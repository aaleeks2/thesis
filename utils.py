import pandas as pd


def find_movies(dataframe: pd.DataFrame, column_name: str, query_string: str):
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the dataframe")

    mask = dataframe[column_name].str.contains(query_string, case=False, na=False)
    matching_movies = dataframe[mask]
    result = matching_movies[[column_name]].reset_index()
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


def get_movies_by_ids(movie_ids):
    movies = pd.read_csv('thesis_datasets/movies.csv')
    found_movies = movies[movies['movieId'].isin(movie_ids)]
    movie_dict = dict(zip(found_movies['movieId'], found_movies['title']))
    return movie_dict


def get_user_ratings(user_id: int):
    movies_data = pd.read_csv('thesis_datasets/movies.csv')
    ratings_data = pd.read_csv('thesis_datasets/ratings.csv')
    merged = pd.merge(ratings_data, movies_data, on='movieId')
    ratings = merged.loc[merged['userId'] == user_id]
    return ratings[['title', 'rating']]
