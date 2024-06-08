import streamlit as st
import thesis
st.set_page_config(layout="wide")
st.title('Content based recommender')
st.write('***')

contentBased = thesis.ContentBasedRecommender()
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Demographic filtering')
    top_n = st.slider('Pick top N films', min_value=3, max_value=100, value=0)
    if top_n != 0:
        st.write(contentBased.get_demographic_filtering(top_n))

with col2:
    st.subheader('Plot description filtering')
    selected_movie = st.selectbox('Select a movie', contentBased.get_movies(), index=None)
    top_n = st.slider('N movies to recommend', min_value=1, max_value=10, value=0)
    if selected_movie is not None and top_n > 0:
        st.write(contentBased.get_plot_description_based_recommendations(selected_movie, top_n))

with col3:
    st.subheader('Credits, genres and keywords filtering')
    selected_movie = st.selectbox(' Select a movie', contentBased.get_movies(), index=None)
    top_n = st.slider(' N movies to recommend', min_value=1, max_value=10, value=0)
    if selected_movie is not None and top_n > 0:
        st.write(contentBased.get_credits_genres_keywords_based_recommendations(selected_movie, top_n))
