import pandas as pd
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from ast import literal_eval
from utils import find_movies

PATH_TMDB_5000_CREDITS = 'thesis_datasets/tmdb_5000_credits.csv'
PATH_TMDB_5000_MOVIES = 'thesis_datasets/tmdb_5000_movies.csv'


class ContentBasedRecommender:
    def __init__(self):
        self._credits: pd.DataFrame = pd.read_csv(PATH_TMDB_5000_CREDITS)
        self._movies_single: pd.DataFrame = pd.read_csv(PATH_TMDB_5000_MOVIES)
        self._credits.columns = ['id', 'tittle', 'cast', 'crew']
        self._movies = self._movies_single.merge(self._credits, on='id')

    def search_for_movie(self, query):
        return self._movies_single[self._movies_single['original_title'] == query]

    def get_movies(self):
        return self._movies['original_title']

    def get_demographic_filtering(self, top_n: int = 10, quantile: int = 0.5):
        """
        IMDB's weighted rating
        :param quantile:
        :param top_n:
        :return:
        """
        start = time.time()
        mean_vote = self._movies['vote_average'].mean()
        minimum_votes = self._movies['vote_count'].quantile(quantile)
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
        time_in_ms = (end - start) * 1000
        print(f'Time taken to create ranking: {round(time_in_ms, 2)} ms.')
        return result

    def get_plot_description_based_recommendations(self, title: str, top_n: int = 5):
        tfidf = TfidfVectorizer(stop_words='english')
        self._movies['overview'] = self._movies['overview'].fillna('')
        tfidf_matrix = tfidf.fit_transform(self._movies['overview'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        indices = pd.Series(self._movies.index, index=self._movies['title']).drop_duplicates()
        return self._get_recommendation(title, ['title', 'overview'], indices, cosine_sim, top_n)

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
        return self._get_recommendation(title, ['title', 'cast', 'keywords', 'genres'],
                                        indices, cosine_sim, top_n)

    def _get_recommendation(self, title: str, result_columns, indices: pd.Series, similarities, top_n: int = 5):
        idx = indices[title]
        sim_scores = list(enumerate(similarities[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]
        movie_indices = [i[0] for i in sim_scores]
        return self._movies[result_columns].iloc[movie_indices]

    def _get_director(self, crew):
        for person in crew:
            if person['job'] == 'Director':
                return person['name']
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
