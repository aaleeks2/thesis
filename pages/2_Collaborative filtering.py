import streamlit as st

import thesis
st.set_page_config(layout="wide")
st.title('Collaborative filtering recommender')
st.write('***')

svd = thesis.CollaborativeBasedRecommender(thesis.AlgorithmType.SVD)
nmf = thesis.CollaborativeBasedRecommender(thesis.AlgorithmType.NMF)
knn = thesis.CollaborativeBasedRecommender(thesis.AlgorithmType.KNN)
col01, col02 = st.columns(2)
titles = svd.get_titles()
with col01:
    st.header('Decomposition models')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('SVD')
        selected_movie = st.selectbox('Select a movie', titles, index=None)
        top_n = st.slider('N movies to recommend', min_value=1, max_value=10, value=0)
        if selected_movie is not None and top_n > 0:
            st.write(svd.get_recommendation(selected_movie, top_n))

    with col2:
        st.subheader('NMF')
        selected_movie = st.selectbox('Select a movie ', titles, index=None)
        top_n = st.slider('N movies to recommend ', min_value=1, max_value=10, value=0)
        if selected_movie is not None and top_n > 0:
            st.write(nmf.get_recommendation(selected_movie, top_n))

with col02:
    st.header('Unsupervised learning recommender - K nearest neighbors')


# with col3:
#     st.subheader('K Nearest Neighbors')
#     selected_movie = st.selectbox('Select a movie ', titles, index=None)
#     top_n = st.slider(' N movies to recommend', min_value=1, max_value=10, value=0)
#     if selected_movie is not None and top_n > 0:
#         st.write(contentBased.get_credits_genres_keywords_based_recommendations(selected_movie, top_n))
