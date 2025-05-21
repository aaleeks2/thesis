import pandas as pd
import streamlit as st
import content_based
import matplotlib.pyplot as plt
from utils import find_movies
from wordcloud import WordCloud

st.set_page_config(layout="wide")
st.title('Content based recommender')
st.write('***')

contentBased = content_based.ContentBasedRecommender()
col1, col11 = st.columns(2)
col2, col22 = st.columns(2)
col3, col33 = st.columns(2)

with col1:
    st.subheader('Demographic filtering')  # default value
    top_n = st.slider('Pick top N films', min_value=3, max_value=100, value=0)

    with st.expander("Options"):
        quantile_slider = st.slider('Quantile', min_value=0.01, max_value=0.99, step=0.01, value=0.50)

    if top_n != 0:
        st.table(contentBased.get_demographic_filtering(top_n, quantile_slider))

with col2:
    st.subheader('Plot description filtering')
    selected_movie_title = st.selectbox('Select a movie', contentBased.get_movies(), index=None)
    the_movie = contentBased.search_for_movie(selected_movie_title)
    overview = the_movie.overview
    if len(overview.values) > 0 is not None:
        st.write(f'Plot overview:\n{overview.values[0]}')
    top_n = st.slider('N movies to recommend', min_value=1, max_value=10, value=0)
    if selected_movie_title is not None and top_n > 0:
        st.table(contentBased.get_plot_description_based_recommendations(selected_movie_title, top_n))

    if contentBased.plot_desc_recommender_plot_data is not None:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
            dict(zip(contentBased.plot_desc_recommender_plot_data['top_features'], contentBased.plot_desc_recommender_plot_data['top_scores'])))

        # Wyświetlenie chmury słów
        xd = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(xd)

        # with regular_plot_col:
        xd2 = plt.figure(figsize=(10, 6))
        plt.barh(contentBased.plot_desc_recommender_plot_data['top_features'], contentBased.plot_desc_recommender_plot_data['top_scores'], color='skyblue')
        plt.xlabel('Waga TF-IDF')
        plt.title('Top 5 słów według wag TF-IDF')
        plt.gca().invert_yaxis()  # Odwrócenie osi Y, aby najważniejsze słowa były na górze
        st.pyplot(xd2)

with col3:
    st.subheader('Credits, genres and keywords filtering')
    selected_movie = st.selectbox(' Select a movie', contentBased.get_movies(), index=None)
    top_n = st.slider(' N movies to recommend', min_value=1, max_value=10, value=0)
    if selected_movie is not None and top_n > 0:
        st.table(contentBased.get_credits_genres_keywords_based_recommendations(selected_movie, top_n))
