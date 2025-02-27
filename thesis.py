import time
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD, NMF
from surprise.model_selection import cross_validate
from ast import literal_eval


def find_movies(dataframe: pd.DataFrame, column_name: str, query_string: str):
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the dataframe")

    mask = dataframe[column_name].str.contains(query_string, case=False, na=False)
    matching_movies = dataframe[mask]
    result = matching_movies[[column_name]].reset_index()
    return result

class ContentBasedRecommender:
    def __init__(self):
        self._credits: pd.DataFrame = pd.read_csv('thesis_datasets/tmdb_5000_credits.csv')
        self._movies: pd.DataFrame = pd.read_csv('thesis_datasets/tmdb_5000_movies.csv')
        self._credits.columns = ['id', 'tittle', 'cast', 'crew']
        self._movies = self._movies.merge(self._credits, on='id')
        print('Content Based constructor')

    def search_for_movie(self, query):
        return find_movies(self._movies, 'original_title', query)

    def get_movies(self):
        return self._movies['original_title']

    def get_demographic_filtering(self, top_n: int = 10):
        """
        IMDB's weighted rating
        :param top_n:
        :return:
        """
        start = time.time()
        mean_vote = self._movies['vote_average'].mean()
        minimum_votes = self._movies['vote_count'].quantile(0.50)
        q_movies = self._movies.copy().loc[self._movies['vote_count'] >= minimum_votes]

        def weighted_rating(x, m=minimum_votes, c=mean_vote):
            v = x['vote_count']
            R = x['vote_average']
            # Calculation based on the IMDB formula
            temp = (v / (v + m))
            return (temp * R) + (temp * c)

        q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
        q_movies = q_movies.sort_values(by='score', ascending=False)
        result = q_movies[['title', 'vote_count', 'vote_average', 'score']].head(top_n)
        end = time.time()
        print(f'Time taken to create a ranking in ms: {end - start}')
        return result

    def get_plot_description_based_recommendations(self, title: str, top_n: int = 5):
        tfidf = TfidfVectorizer(stop_words='english')
        self._movies['overview'] = self._movies['overview'].fillna('')
        tfidf_matrix = tfidf.fit_transform(self._movies['overview'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        indices = pd.Series(self._movies.index, index=self._movies['title']).drop_duplicates()
        return self._get_recommendation(title, indices, cosine_sim, top_n)

    def get_credits_genres_keywords_based_recommendations(self, title: str, top_n: int = 5):
        features = ['cast', 'crew', 'keywords', 'genres']
        for feature in features:
            self._movies[feature] = self._movies[feature].apply(literal_eval)
        self._movies['director'] = self._movies['crew'].apply(self._get_director)
        features = ['cast', 'keywords', 'genres']
        for feature in features:
            self._movies[feature] = self._movies[feature].apply(self._get_list)
        features = ['cast', 'keywords', 'director', 'genres']
        for feature in features:
            self._movies[feature] = self._movies[feature].apply(self._clean_data)
        self._movies['soup'] = self._movies.apply(self._create_soup, axis=1)
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(self._movies['soup'])
        cosine_sim = cosine_similarity(count_matrix, count_matrix)
        self._movies = self._movies.reset_index()
        indices = pd.Series(self._movies.index, index=self._movies['title'])
        return self._get_recommendation(title, indices, cosine_sim, top_n)

    def _get_recommendation(self, title: str, indices: pd.Series, cosine_sim, top_n: int = 5):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]
        movie_indices = [i[0] for i in sim_scores]
        return self._movies['title'].iloc[movie_indices]

    def _get_director(self, x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return None

    def _get_list(self, x):
        if isinstance(x, list):
            names = [i['name'] for i in x]
            if len(names) > 3:
                names = names[:3]
            return names
        return []

    def _clean_data(self, x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            # Check if director exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''

    def _create_soup(self, x):
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
