import streamlit as st
import pandas as pd
import collaborative
import utils
st.set_page_config(layout="wide")
st.title('Collaborative filtering recommender')
st.write('***')

collab_recommenders = collaborative.CollaborativeBasedRecommender()
previous_user_id = 0
titles = collab_recommenders.get_movie_titles()
nn_algorithms = ['ball_tree', 'kd_tree', 'brute']
brute_metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
ball_tree_metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'dice', 'euclidean', 'hamming', 'infinity', 'jaccard', 'l1', 'l2', 'manhattan', 'minkowski', 'p', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath']
kd_tree_metrics = ['chebyshev', 'cityblock', 'euclidean', 'infinity', 'l1', 'l2', 'manhattan', 'minkowski', 'p']
metrics = {'brute': brute_metrics, 'ball_tree': ball_tree_metrics, 'kd_tree': kd_tree_metrics}


if "user_id" not in st.session_state:
    st.session_state.user_id = 0


def selected_user_id():
    st.write(f'Selected user ID: {st.session_state.user_id}')


def set_new_user_id(new_user_id: int):
    st.session_state.user_id = new_user_id


def add_rating(movie, user_rating):
    the_user_id = st.session_state.user_id
    if the_user_id is 'None':
        return st.error('Please select a user ID')

    if movie is None or user_rating is None:
        raise ValueError(f'Missing data: movie: {movie}, rating: {user_rating}, user ID: {the_user_id}')
    utils.add_rating(the_user_id, movie, user_rating)
    global previous_user_id
    previous_user_id = the_user_id


st.header('User section')
col11, col12 = st.columns(2)
with col11:
    col111, col112 = st.columns(2)
    with col111:
        selected_user_id()
    with col112:
        add_user_button = st.button(label='Get new user ID', type='primary')
        if add_user_button:
            set_new_user_id(utils.get_new_user_id())
            st.rerun()

    st.write("Find user")
    select_user_col, button_col = st.columns(2)
    with select_user_col:
        selected_user_id = st.selectbox('Select a user', collab_recommenders.knn.get_user_ids(), index=None,
                                        label_visibility="collapsed")
    with button_col:
        find_user_button = st.button(label='Find user', type='primary')
    if selected_user_id and find_user_button:
        set_new_user_id(selected_user_id)
        st.rerun()

    if st.session_state.user_id is not 0 and st.session_state.user_id is not None:
        user_ratings = utils.get_user_ratings(st.session_state.user_id)
        if len(user_ratings) == 0:
            st.write('User has no ratings')
        else:
            st.write(user_ratings)

with col12:
    with st.expander("Add rating"):
        with st.form('add_rating_form', clear_on_submit=True):
            ar_col_1, ar_col_2 = st.columns(2)
            with ar_col_1:
                selected_movie = st.selectbox('Select a movie', titles, index=None, key='movie')
            with ar_col_2:
                rating = st.slider(label='Rating', min_value=1.0, max_value=5.0, step=0.5, key='rating')
            submit_button = st.form_submit_button(label='Submit')
            if submit_button:
                add_rating(selected_movie, rating)
                st.rerun()


st.write('***')
st.header('Recommenders')

col01, col02 = st.columns(2)
with col01:
    st.header('Explicit collaborative filtering - Decomposition models')
    st.subheader('SVD')
    if st.session_state.user_id is not 0 and st.session_state.user_id is not None:
        top_n = st.slider('N movies to recommend', min_value=1, max_value=10, value=0)
        recommend_button = st.button('Recommend with SVD')
        if recommend_button and top_n > 0:
            collab_recommenders.decomposition_algo.set_algo(collaborative.AlgorithmType.SVD)
            st.write(pd.DataFrame(collab_recommenders.decomposition_algo.get_recommendations(st.session_state.user_id,
                                                                                             top_n)))
    else:
        st.write('Select user to recommend a movie')

    st.subheader('NMF')
    if st.session_state.user_id is not 0 and st.session_state.user_id is not None:
        top_n = st.slider('N movies to recommend ', min_value=1, max_value=10, value=0)
        recommend_button = st.button('Recommend with NMF')
        if recommend_button and top_n > 0:
            collab_recommenders.decomposition_algo.set_algo(collaborative.AlgorithmType.NMF)
            st.write(pd.DataFrame(collab_recommenders.decomposition_algo.get_recommendations(st.session_state.user_id,
                                                                                             top_n)))
    else:
        st.write('Select user to recommend a movie')

with col02:
    st.header('Implicit collaborative filtering - unsupervised learning recommender - K nearest neighbors')
    selected_movie = st.selectbox('Select a movie ', titles, index=None)
    top_n = st.slider('N movies to recommend  ', min_value=1, max_value=10, value=0)
    algorithm = st.selectbox('Select an algorithm ', nn_algorithms, index=None)
    if algorithm:
        metrics_fixed = metrics[algorithm]
        metric = st.selectbox('Select a metric', metrics_fixed, index=None)
        recommend_button = st.button('Recommend with KNN')
        if recommend_button and top_n > 0 and selected_movie is not None:
            collab_recommenders.set_knn(metric, algorithm)
            st.write(pd.DataFrame(collab_recommenders.knn.get_n_recommendations(selected_movie, top_n)))
