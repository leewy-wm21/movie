import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Load dataset and similarity matrix
df = pd.read_csv('data.csv')
titles = df['title'].values

# Feature extraction and similarity calculation
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()
similarity = cosine_similarity(vectors)

# API key for TMDB
API_KEY_AUTH = "b8c96e534866701532768a313b978c8b"

# Function to fetch movie posters from TMDB
def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY_AUTH}')
    data = response.json()
    poster_path = data['poster_path']
    full_path = 'https://image.tmdb.org/t/p/w500/' + poster_path
    return full_path

# Function to recommend movies based on a given movie
def recommender(movie):
    movie_index = df[df['title'] == movie].index[0]
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:21]
    movie_recommend = []
    movie_recommend_posters = []
    for i in movies_list:
        movie_id = df.iloc[i[0]]['movie_id']
        movie_recommend.append(df.iloc[i[0]]['title'])
        movie_recommend_posters.append(fetch_poster(movie_id))
    return movie_recommend, movie_recommend_posters

# Setting up Streamlit app
st.set_page_config(layout="wide")
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# App title and movie selection
st.title('CINEPHILE ENGINE')
selected_movie = st.selectbox('Type a Movie', options=titles)

# Button to trigger recommendation
if st.button('Recommend'):
    recommended_movie_names, recommended_movie_posters = recommender(selected_movie)

    # Display recommendations in a grid
    num_recommendations = len(recommended_movie_names)
    cols_per_row = 5
    rows = (num_recommendations + cols_per_row - 1) // cols_per_row

    for row in range(rows):
        cols = st.columns(cols_per_row)
        for i in range(cols_per_row):
            index = row * cols_per_row + i
            if index < num_recommendations:
                with cols[i]:
                    st.text(recommended_movie_names[index])
                    st.image(recommended_movie_posters[index])
