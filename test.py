import collaborative
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn import neighbors

if __name__ == '__main__':
    # ratings = pd.read_csv('thesis_datasets/ratings.csv')
    # movies = pd.read_csv('thesis_datasets/movies.csv')
    # movies_data = movies[['movieId', 'title']]
    # ratings_data = ratings[['userId', 'movieId', 'rating']]
    # merged = pd.merge(ratings_data, movies_data, on='movieId')
    #
    # user_movie_table = merged.pivot_table(index=['title'], columns=['userId'],
    #                    values='rating').fillna(0)
    # print(user_movie_table.head(10))
    # user_movie_table_matrix = csr_matrix(user_movie_table)  # [[movie, user, rating], ...]
    # normalized = Normalizer().fit_transform(user_movie_table_matrix)
    # xd = user_movie_table.iloc[9, :].values.reshape(1, -1)
    # print(xd)
    # print(normalized[0:10])
    print(sorted(neighbors.VALID_METRICS['kd_tree']))

