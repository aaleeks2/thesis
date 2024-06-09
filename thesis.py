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


class AlgorithmType(Enum):
    SVD = 0
    NMF = 1
    KNN = 3


class CollaborativeBasedRecommender:
    DECOMPOSITION_MODELS = [AlgorithmType.SVD, AlgorithmType.NMF]

    def __init__(self, algo_type: AlgorithmType = AlgorithmType.SVD):
        self._ratings = pd.read_csv('thesis_datasets/ratings.csv')
        self._movies = pd.read_csv('thesis_datasets/movies.csv')
        self._reader = Reader()
        self._algo_type = algo_type
        if algo_type in self.DECOMPOSITION_MODELS:
            self._decomposition_data = Dataset.load_from_df(self._ratings[['userId', 'movieId', 'rating']], self._reader)
        if algo_type is AlgorithmType.KNN:
            self._knn_data = self._prepare_knn_data()
        self._trained_model = self._train_model()

    def search_for_movie(self, query: str):
        return find_movies(self._movies, 'title', query)

    def get_titles(self):
        return self._movies['title'].values

    def get_user_ids(self):
        return self._ratings['userId'].astype(int).unique()

    def get_user_ratings(self, user_id: int):
        movies_data = self._movies[['movieId', 'title']]
        ratings_data = self._ratings[['userId', 'movieId', 'rating']]
        merged = pd.merge(ratings_data, movies_data, on='movieId')
        ratings = merged.loc[merged['userId'] == user_id]
        return ratings[['title', 'rating']]

    def _prepare_knn_data(self):
        movies_data = self._movies[['movieId', 'title']]
        ratings_data = self._ratings[['userId', 'movieId', 'rating']]
        merged = pd.merge(ratings_data, movies_data, on='movieId')
        self._merged_knn_data_model = merged
        return merged.pivot_table(index=['title'], columns=['userId'], values='rating').fillna(0)

    def _train_model(self):
        start = time.time()
        print('Model training starts...')
        if self._algo_type in self.DECOMPOSITION_MODELS:
            algo = SVD(n_epochs=5) if self._algo_type == AlgorithmType.SVD else NMF(n_epochs=5)
            cross_validate(algo, self._decomposition_data, measures=['RMSE', 'MAE'], cv=5)
            print(f'Model training time: {round(time.time() - start, 3)} sec')
            return algo

        user_movie_table_matrix = csr_matrix(self._knn_data.values)
        self._csr_matrix_data = user_movie_table_matrix
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        model_knn.fit(user_movie_table_matrix)
        print(f'Model training time: {round(time.time() - start, 3)} sec')
        return model_knn

    def get_recommendation(self, query: int | str, top_n: int = 5, r_id: int = 4, cv: int = 5):
        print(f'Getting recommendations for: {query}...')
        start_time = time.time()
        print('Prediction starts...')
        result = []
        if self._algo_type in self.DECOMPOSITION_MODELS:
            movie_ids = self._ratings["movieId"].unique()
            movie_ids_user = self._ratings.loc[self._ratings["userId"] == query, "movieId"]
            movie_ids_to_pred = np.setdiff1d(movie_ids, movie_ids_user)
            test_set = [[query, movie_id, r_id] for movie_id in movie_ids_to_pred]
            predictions = self._trained_model.test(test_set)
            pred_ratings = np.array([pred.est for pred in predictions])
            index_max = (-pred_ratings).argsort()[:top_n]
            for i in index_max:
                movie_id = movie_ids_to_pred[i]
                result.append(self._movies[self._movies["movieId"] == movie_id]["title"].values[0])
                print(self._movies[self._movies["movieId"] == movie_id]["title"].values[0], pred_ratings[i])
        else:
            movie_title = query
            query_index_2 = self._knn_data.index.values.tolist().index(movie_title)
            distances, indices = self._trained_model.kneighbors(self._knn_data.iloc[query_index_2, :].values.reshape(1, -1), n_neighbors=top_n+1)
            recommended_movies = []
            calculated_distances = []
            for i in range(1, len(indices[0])):
                recommended_movies.append(self._knn_data.index[indices.flatten()][i])
                calculated_distances.append(distances.flatten()[i])

            movie_series = pd.Series(recommended_movies, name='movie')
            distance_series = pd.Series(calculated_distances, name='distance')
            merged_series = pd.concat([movie_series, distance_series], axis=1)
            sorted_merged_series = merged_series.sort_values('distance', ascending=True)

            for index, row in sorted_merged_series.iterrows():
                result.append(row["movie"])

        print(f'Prediction finished in {round(time.time() - start_time, 2)} sec')
        return result


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
