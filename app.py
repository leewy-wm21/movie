import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Load data (assumes your CSV data file is named 'data.csv')
df = pd.read_csv('data.csv')
titles = df['title'].values
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()
similarity = cosine_similarity(vectors)

API_KEY_AUTH = "b8c96e534866701532768a313b978c8b"


def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY_AUTH}')
    data = response.json()
    poster_path = data['poster_path']
    full_path = 'https://image.tmdb.org/t/p/w500/' + poster_path
    return full_path


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


# Streamlit app configuration
st.set_page_config(layout="wide")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

st.title('CINEPHILE ENGINE')
selected_movie = st.selectbox('Type a Movie', options=titles)

if st.button('Recommend'):
    recommended_movie_names, recommended_movie_posters = recommender(selected_movie)

    # Display recommended movies in a grid
    num_columns = 5  # Adjust this to change the number of columns
    for i in range(0, len(recommended_movie_names), num_columns):
        cols = st.columns(num_columns)  # Create a row with `num_columns` columns
        for j in range(num_columns):
            if i + j < len(recommended_movie_names):  # Check if there's a movie to display
                cols[j].text(recommended_movie_names[i + j])
                cols[j].image(recommended_movie_posters[i + j], use_column_width=True)
