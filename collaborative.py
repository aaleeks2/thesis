from enum import Enum
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, accuracy, SVD, NMF
from surprise.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import Normalizer
import time

class AlgorithmType(Enum):
    SVD = 0
    NMF = 1
    KNN = 3


class CollaborativeBasedRecommender:
    def __init__(self):
        self._ratings = pd.read_csv('thesis_datasets/ratings.csv')
        self._movies = pd.read_csv('thesis_datasets/movies.csv')
        self.decomposition_algo = CollaborativeDecomposition(self._ratings)
        self.knn = CollaborativeKnn()

    def get_movie_titles(self):
        return self.knn.get_titles()

    def set_knn(self, metric: str, algorith: str):
        self.knn = CollaborativeKnn(metric, algorith)


class CollaborativeDecomposition:
    def __init__(self, ratings: pd.DataFrame):
        self._ratings = pd.read_csv('thesis_datasets/ratings.csv')
        self._movies = pd.read_csv('thesis_datasets/movies.csv')
        self._reader = Reader(rating_scale=(0.5, 5))
        self._data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], self._reader)
        self._algo = None
        self._movies_dict = dict(zip(self._movies['movieId'], self._movies['title']))

    def set_algo(self, algo_type: AlgorithmType = AlgorithmType.SVD):
        if algo_type == AlgorithmType.SVD:
            self._algo = SVD()
        else:
            self._algo = NMF()

    def validate_model(self, evaluate: bool = False):
        train_set, test_set = train_test_split(self._data, test_size=0.2)

        if self._algo is None:
            self.set_algo()

        start = time.time()
        self._algo.fit(train_set)
        end = time.time()
        print(f'Algorithm trained in {end - start:.2f} seconds')
        predictions = self._algo.test(test_set)
        if evaluate:
            rmse = accuracy.rmse(predictions, verbose=True)
            validation_results = cross_validate(self._algo, self._data, measures=['rmse', 'mae', 'mse', 'fcp'], cv=5,
                                                verbose=True, return_train_measures=True)
            return validation_results, rmse

    def get_recommendations(self, user_id: int, recommendations_number: int):
        start = time.time()
        self.validate_model()
        user_predictions = []

        for index, row in self._movies.iterrows():
            movie_ratings = self._ratings[self._ratings['movieId'] == row['movieId']]
            user_movie_ratings = movie_ratings[movie_ratings['userId'] == user_id]
            if user_movie_ratings.empty:
                prediction = self._algo.predict(user_id, row['movieId'])
            else:
                prediction = self._algo.predict(user_id, row['movieId'], user_movie_ratings.iloc[0, 2])
            user_predictions.append(prediction)

        user_predictions.sort(key=lambda x: x.est, reverse=True)
        top_n_recommendations = user_predictions[:recommendations_number]

        described_predictions = []
        for pred in top_n_recommendations:
            described_predictions.append({'title': self._movies_dict.get(pred.iid), 'est_rating': round(pred.est, 2)})

        end = time.time()
        prediction_time = round((end - start), 2)
        print(f'Prediction time: {prediction_time}')
        return described_predictions


class CollaborativeKnn:
    def __init__(self, metric: str = 'cosine', algorithm: str = 'brute'):
        self._ratings = pd.read_csv('thesis_datasets/ratings.csv')
        self._movies = pd.read_csv('thesis_datasets/movies.csv')
        movies_data = self._movies[['movieId', 'title']]
        ratings_data = self._ratings[['userId', 'movieId', 'rating']]
        merged = pd.merge(ratings_data, movies_data, on='movieId')
        self._merged_knn_data_model = merged
        self._metric = metric
        self._algorithm = algorithm
        self._user_movie_table = merged.pivot_table(index=['title'], columns=['userId'],
                                                    values='rating').fillna(0)
        self._algo = None
        self._validation_results = None

    def train_model(self):
        if self._algo is 'brute':
            user_movie_table_matrix = csr_matrix(self._user_movie_table)
            normalized_user_movie_table_matrix = Normalizer().fit_transform(user_movie_table_matrix)
            data = normalized_user_movie_table_matrix
        else:
            data = self._user_movie_table
        model_knn = NearestNeighbors(metric=self._metric, algorithm=self._algorithm)
        model_knn.fit(data)
        self._algo = model_knn

    def get_titles(self):
        return self._movies['title'].values

    def get_user_ids(self):
        return self._ratings['userId'].astype(int).unique()

    def get_n_recommendations(self, movie_title: str, n_recommendations: int = 5):
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
