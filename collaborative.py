from enum import Enum
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, accuracy, SVD, NMF
from surprise.model_selection import cross_validate, train_test_split
from utils import get_movies_by_ids


class AlgorithmType(Enum):
    SVD = 0
    NMF = 1
    KNN = 3


class CollaborativeBasedRecommender:
    def __init__(self, algo_type: AlgorithmType = AlgorithmType.SVD):
        self._ratings = pd.read_csv('thesis_datasets/ratings.csv')
        self._movies = pd.read_csv('thesis_datasets/movies.csv')
        self.decomposition_algo = CollaborativeDecomposition(self._ratings, algo_type)
        self.knn = CollaborativeKnn()

    def get_movie_titles(self):
        return self.knn.get_titles()


class CollaborativeDecomposition:
    def __init__(self, ratings: pd.DataFrame, algo_type: AlgorithmType = AlgorithmType.SVD):
        self._reader = Reader(rating_scale=(0.5, 5))
        self._data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], self._reader)
        self._algo = None
        self._algo_type = algo_type
        self._validation_results = None
        self._rmse = None

    def set_algo_type(self, algo_type: AlgorithmType):
        self._algo_type = algo_type

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
        movies_dict = get_movies_by_ids(top_n_recommendations_movie_ids)
        result = []
        for pred in top_n_recommendations:
            print(f"Movie [ID: {pred.iid}] {movies_dict.get(pred.iid)}, Estimated Rating: {pred.est:.2f}")
            entry = {'title': movies_dict.get(pred.iid), 'est_rating': round(pred.est, 2)}
            result.append(entry)

        return result


class CollaborativeKnn:
    def __init__(self):
        self._ratings = pd.read_csv('thesis_datasets/ratings.csv')
        self._movies = pd.read_csv('thesis_datasets/movies.csv')
        movies_data = self._movies[['movieId', 'title']]
        ratings_data = self._ratings[['userId', 'movieId', 'rating']]
        merged = pd.merge(ratings_data, movies_data, on='movieId')
        self._merged_knn_data_model = merged
        self._user_movie_table = merged.pivot_table(index=['title'], columns=['userId'],
                                                    values='rating').fillna(0)
        self._algo = None
        self._validation_results = None

    def train_model(self):
        user_movie_table_matrix = csr_matrix(self._user_movie_table)
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        model_knn.fit(user_movie_table_matrix)
        self._algo = model_knn

    def get_titles(self):
        return self._movies['title'].values

    def get_user_ids(self):
        return self._ratings['userId'].astype(int).unique()

    def get_n_recommendations(self, movie_title: str, n_recommendations: int = 5):
        if self._algo is None:
            self.train_model()

        query_index = self._user_movie_table.index.get_loc(movie_title)
        distances, indices = self._algo.kneighbors(self._user_movie_table.iloc[query_index, :].values.reshape(1, -1),
                                                   n_neighbors=n_recommendations + 1)
        recommended_movies = []
        calculated_distances = []

        for i in range(1, len(distances.flatten())):
            recommended_movies.append(self._user_movie_table.index[indices.flatten()][i])
            calculated_distances.append(distances.flatten()[i])

        movie_series = pd.Series(recommended_movies, name='movie')
        distance_series = pd.Series(calculated_distances, name='distance')
        merged_series = pd.concat([movie_series, distance_series], axis=1)
        sorted_merged_series = merged_series.sort_values('distance', ascending=True)

        result = []
        for index, row in sorted_merged_series.iterrows():
            entry = {'title': row["movie"], 'dist': round(row["distance"], 2)}
            result.append(entry)

        return result