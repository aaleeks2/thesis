import time
from enum import Enum
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, accuracy, SVD, NMF
from surprise.model_selection import cross_validate, train_test_split
from utils import find_movies


class AlgorithmType(Enum):
    SVD = 0
    NMF = 1
    KNN = 3


class CollaborativeBasedRecommender:
    DECOMPOSITION_MODELS = [AlgorithmType.SVD, AlgorithmType.NMF]

    def __init__(self, algo_type: AlgorithmType = AlgorithmType.SVD):
        self._ratings = pd.read_csv('thesis_datasets/ratings.csv')
        self._movies = pd.read_csv('thesis_datasets/movies.csv')
        self.decomposition_algo = CollaborativeDecomposition(self._ratings, algo_type)
        self.knn = CollaborativeKnn()
        # self._reader = Reader()
        # self._algo_type = algo_type
        # if algo_type in self.DECOMPOSITION_MODELS:
        #     self._decomposition_data = Dataset.load_from_df(self._ratings[['userId', 'movieId', 'rating']], self._reader)
        # if algo_type is AlgorithmType.KNN:
        #     self._knn_data = self._prepare_knn_data()
        # self._trained_model = self._train_model()

    def get_movies_by_ids(self, movie_ids):
        movies = self._movies[self._movies['movieId'].isin(movie_ids)]
        movie_dict = dict(zip(movies['movieId'], movies['title']))
        return movie_dict

    # def search_for_movie(self, query: str):
    #     return find_movies(self._movies, 'title', query)
    #
    # def get_titles(self):
    #     return self._movies['title'].values
    #
    # def get_user_ids(self):
    #     return self._ratings['userId'].astype(int).unique()
    #
    # def get_user_ratings(self, user_id: int):
    #     movies_data = self._movies[['movieId', 'title']]
    #     ratings_data = self._ratings[['userId', 'movieId', 'rating']]
    #     merged = pd.merge(ratings_data, movies_data, on='movieId')
    #     ratings = merged.loc[merged['userId'] == user_id]
    #     return ratings[['title', 'rating']]
    #
    # def _prepare_knn_data(self):
    #     movies_data = self._movies[['movieId', 'title']]
    #     ratings_data = self._ratings[['userId', 'movieId', 'rating']]
    #     merged = pd.merge(ratings_data, movies_data, on='movieId')
    #     self._merged_knn_data_model = merged
    #     return merged.pivot_table(index=['title'], columns=['userId'], values='rating').fillna(0)
    #
    # def _train_model(self):
    #     start = time.time()
    #     print('Model training starts...')
    #     if self._algo_type in self.DECOMPOSITION_MODELS:
    #         algo = SVD(n_epochs=5) if self._algo_type == AlgorithmType.SVD else NMF(n_epochs=5)
    #         cross_validate(algo, self._decomposition_data, measures=['RMSE', 'MAE'], cv=5)
    #         print(f'Model training time: {round(time.time() - start, 3)} sec')
    #         return algo
    #
    #     user_movie_table_matrix = csr_matrix(self._knn_data.values)
    #     self._csr_matrix_data = user_movie_table_matrix
    #     model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    #     model_knn.fit(user_movie_table_matrix)
    #     print(f'Model training time: {round(time.time() - start, 3)} sec')
    #     print(model_knn.effective_metric_)
    #     return model_knn
    #
    # def get_recommendation(self, query: int | str, top_n: int = 5, r_id: int = 4, cv: int = 5):
    #     print(f'Getting recommendations for: {query}...')
    #     start_time = time.time()
    #     print('Prediction starts...')
    #     result = []
    #     if self._algo_type in self.DECOMPOSITION_MODELS:
    #         movie_ids = self._ratings["movieId"].unique()
    #         movie_ids_user = self._ratings.loc[self._ratings["userId"] == query, "movieId"]
    #         movie_ids_to_pred = np.setdiff1d(movie_ids, movie_ids_user)
    #         test_set = [[query, movie_id, r_id] for movie_id in movie_ids_to_pred]
    #         predictions = self._trained_model.test(test_set)
    #         pred_ratings = np.array([pred.est for pred in predictions])
    #         index_max = (-pred_ratings).argsort()[:top_n]
    #         for i in index_max:
    #             movie_id = movie_ids_to_pred[i]
    #             result.append(self._movies[self._movies["movieId"] == movie_id]["title"].values[0])
    #             print(self._movies[self._movies["movieId"] == movie_id]["title"].values[0], pred_ratings[i])
    #     else:
    #         movie_title = query
    #         query_index_2 = self._knn_data.index.values.tolist().index(movie_title)
    #         distances, indices = self._trained_model.kneighbors(self._knn_data.iloc[query_index_2, :].values.reshape(1, -1), n_neighbors=top_n+1)
    #         recommended_movies = []
    #         calculated_distances = []
    #         for i in range(1, len(indices[0])):
    #             recommended_movies.append(self._knn_data.index[indices.flatten()][i])
    #             calculated_distances.append(distances.flatten()[i])
    #
    #         movie_series = pd.Series(recommended_movies, name='movie')
    #         distance_series = pd.Series(calculated_distances, name='distance')
    #         merged_series = pd.concat([movie_series, distance_series], axis=1)
    #         sorted_merged_series = merged_series.sort_values('distance', ascending=True)
    #
    #         for index, row in sorted_merged_series.iterrows():
    #             result.append(row["movie"])
    #
    #     print(f'Prediction finished in {round(time.time() - start_time, 2)} sec')
    #     return result

class CollaborativeDecomposition:
    def __init__(self, ratings: pd.DataFrame, algo_type: AlgorithmType = AlgorithmType.SVD):
        self._reader = Reader(rating_scale=(0.5, 5))
        self._data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], self._reader)
        self._algo = None
        self._algo_type = algo_type
        self._validation_results = None
        self._rmse = None

    def validate_model(self):
        train_set, test_set = train_test_split(self._data, test_size=0.2)

        if self._algo is None:
            if self._algo_type == AlgorithmType.SVD:
                algo = SVD()
            else:
                algo = NMF()
            algo.fit(train_set)
            self._algo = algo

        predictions = self._algo.test(test_set)
        self._rmse = accuracy.rmse(predictions, verbose=True)
        self._validation_results = cross_validate(self._algo, self._data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    def get_recommendations(self, user_id: int, recommendations_number: int):
        self.validate_model()
        train_set = self._data.build_full_trainset()
        test_set = train_set.build_anti_testset()
        predictions = self._algo.test(test_set)
        user_predictions = [pred for pred in predictions if pred.uid == user_id]
        user_predictions.sort(key=lambda x: x.est, reverse=True)
        top_n_recommendations = user_predictions[:recommendations_number]
        top_n_recommendations_movie_ids = [pred.iid for pred in top_n_recommendations]
        movies_dict = CollaborativeBasedRecommender().get_movies_by_ids(top_n_recommendations_movie_ids)
        for pred in top_n_recommendations:
            print(f"Movie [ID: {pred.iid}] {movies_dict.get(pred.iid)}, Estimated Rating: {pred.est:.2f}")

        return top_n_recommendations


class CollaborativeKnn:
    def __init__(self):
        self._ratings = pd.read_csv('thesis_datasets/ratings.csv')
        self._movies = pd.read_csv('thesis_datasets/movies.csv')
        movies_data = self._movies[['movieId', 'title']]
        ratings_data = self._ratings[['userId', 'movieId', 'rating']]
        merged = pd.merge(ratings_data, movies_data, on='movieId')
        self._merged_knn_data_model = merged
        self._user_movie_table = merged.pivot_table(index=['title'], columns=['userId'], values='rating').fillna(0)
        self._algo = None

    def train_model(self):
        user_movie_table_matrix = csr_matrix(self._user_movie_table)
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        model_knn.fit(user_movie_table_matrix)
        print(model_knn.effective_metric_)

    def get_titles(self):
        return self._movies['title'].values

    def get_user_ids(self):
        return self._ratings['userId'].astype(int).unique()

def add_rating(user_id: int, movie_title: str, rating: float):
    movies = pd.read_csv('thesis_datasets/movies.csv')
    movie = movies[movies['title'] == movie_title].values
    print(movie)
    if len(movie) == 0:
        new_movie_id = movies.loc[movies['movieId'].idxmax()]
        movie_id = new_movie_id.movieId
        new_movie_entry = {'movieId': movie_id, 'title': movie_title}
        new_movies = movies._append(new_movie_entry, ignore_index=True)
        new_movies.to_csv('thesis_datasets/movies.csv', index=False)
    else:
        movie_id = movie[0][0]

    ratings = pd.read_csv('thesis_datasets/ratings.csv')
    rating_row_index = ratings.loc[(ratings['movieId'] == movie_id) & (ratings['userId'] == user_id)].index
    if len(rating_row_index) == 0:
        ratings.loc[len(ratings.index)] = [user_id, movie_id, rating, 0]
    else:
        the_index = rating_row_index[0]
        ratings.loc[the_index, 'rating'] = rating
    ratings.to_csv('thesis_datasets/ratings.csv', index=False)


def get_new_user_id():
    ratings = pd.read_csv('thesis_datasets/ratings.csv')
    max_user_id = ratings.loc[ratings['userId'].idxmax()].userId
    return int(max_user_id + 1)