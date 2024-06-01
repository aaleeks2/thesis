from enum import Enum
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD, NMF
from surprise.model_selection import cross_validate
from ast import literal_eval

# Content based
# Collaborative filtering


class ContentBasedRecommender:
    def __init__(self):
        self._credits: pd.DataFrame = pd.read_csv('thesis_datasets/tmdb_5000_credits.csv')
        self._movies: pd.DataFrame = pd.read_csv('thesis_datasets/tmdb_5000_movies.csv')
        self._credits.columns = ['id', 'tittle', 'cast', 'crew']
        self._movies = self._movies.merge(self._credits, on='id')

    def get_demographic_filtering(self, top_n: int = 10):
        """
        IMDB's weighted rating
        :param top_n:
        :return:
        """
        mean_vote = self._movies['vote_average'].mean()
        minimum_votes = self._movies['vote_count'].quantile(0.9)
        q_movies = self._movies.copy().loc[self._movies['vote_count'] >= minimum_votes]

        def weighted_rating(x, m=minimum_votes, c=mean_vote):
            v = x['vote_count']
            R = x['vote_average']
            # Calculation based on the IMDB formula
            temp = (v / (v + m))
            return (temp * R) + (temp * c)

        q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
        q_movies = q_movies.sort_values(by='score', ascending=False)
        return q_movies[['title', 'vote_count', 'vote_average', 'score']].head(top_n)

    def get_plot_description_based_recommendations(self, title: str, top_n: int = 5):
        tfidf = TfidfVectorizer(stop_words='english')
        self._movies['overview'] = self._movies['overview'].fillna('')
        tfidf_matrix = tfidf.fit_transform(self._movies['overview'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
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
            # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
            if len(names) > 3:
                names = names[:3]
            return names

        # Return empty list in case of missing/malformed data
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


class AlgorithmType(Enum):
    SVD = 0
    NMF = 1
    KNN = 3


class CollaborativeBasedRecommender:
    def __init__(self, algo_type: AlgorithmType = AlgorithmType.SVD):
        self._ratings = pd.read_csv('thesis_datasets/rating.csv')
        print(len(self._ratings))
        self._movies = pd.read_csv('thesis_datasets/movie.csv')
        self._reader = Reader()
        self._data = Dataset.load_from_df(self._ratings[['userId', 'movieId', 'rating']], self._reader)
        self._algorithm_orchestrator = {
            AlgorithmType.SVD: self._prepare_svd_model,
            AlgorithmType.NMF: self._prepare_nfm_model,
            AlgorithmType.KNN: self._prepare_knn_model
        }
        self._trained_model = self._trained_model(self._algorithm_orchestrator[algo_type]())

    def _prepare_svd_model(self):
        return SVD(n_epochs=5)

    def _prepare_nfm_model(self):
        return NMF(n_epochs=5)

    def _prepare_knn_model(self):
        return NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)

    def _train_model(self, algo):
        return None
        # set model training here

    def predict_rating(self, uid: int, mid: int,  cv: int = 5):
        svd = SVD(n_epochs=10)
        cross_validate(svd, self._data, measures=['RMSE', 'MAE'], cv=cv, verbose=True)
        train_set = self._data.build_full_trainset()
        svd.fit(train_set)
        return svd.predict(uid, mid, 4)

    def get_recommendation_svd(self, user_id: int, top_n: int = 5, r_id: int = 4, cv: int = 5):
        svd = SVD(n_epochs=10)
        cross_validate(svd, self._data, measures=['RMSE', 'MAE'], cv=cv, verbose=True)

        movie_ids = self._ratings["movieId"].unique()
        movie_ids_user = self._ratings.loc[self._ratings["userId"] == user_id, "movieId"]
        movie_ids_to_pred = np.setdiff1d(movie_ids, movie_ids_user)

        test_set = [[user_id, movie_id, r_id] for movie_id in movie_ids_to_pred]
        predictions = svd.test(test_set)
        pred_ratings = np.array([pred.est for pred in predictions])
        index_max = (-pred_ratings).argsort()[:top_n]
        result = []
        for i in index_max:
            movie_id = movie_ids_to_pred[i]
            result.append(self._movies[self._movies["movieId"] == movie_id]["title"].values[0])
            print(self._movies[self._movies["movieId"] == movie_id]["title"].values[0], pred_ratings[i])
        return result

    def get_recommendation_nmf(self, user_id: int, top_n: int = 5, r_id: int = 4, cv: int = 5):
        algo = NMF(n_epochs=10)
        cross_validate(algo, self._data, measures=['RMSE', 'MAE'], cv=cv, verbose=True)

        movie_ids = self._ratings["movieId"].unique()
        movie_ids_user = self._ratings.loc[self._ratings["userId"] == user_id, "movieId"]
        movie_ids_to_pred = np.setdiff1d(movie_ids, movie_ids_user)

        test_set = [[user_id, movie_id, r_id] for movie_id in movie_ids_to_pred]
        predictions = algo.test(test_set)
        pred_ratings = np.array([pred.est for pred in predictions])
        index_max = (-pred_ratings).argsort()[:top_n]
        result = []
        for i in index_max:
            movie_id = movie_ids_to_pred[i]
            result.append(self._movies[self._movies["movieId"] == movie_id]["title"].values[0])
            print(self._movies[self._movies["movieId"] == movie_id]["title"].values[0], pred_ratings[i])
        return result

    # def get_recommendation_knn(self):


cbr = CollaborativeBasedRecommender()
print(cbr.get_recommendation_nfm(3))
