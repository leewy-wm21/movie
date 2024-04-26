import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Load the data
df = pd.read_csv('data.csv')
titles = df['title'].values
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()
similarity = cosine_similarity(vectors)

# API key for TMDB
API_KEY_AUTH = "b8c96e534866701532768a313b978c8b"

# Function to fetch poster
def fetch_poster(movie_id):
    response = requests.get(
        f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY_AUTH}'
    )
    data = response.json()
    poster_path = data.get('poster_path', '')
    if not poster_path:
        return ''
    full_path = 'https://image.tmdb.org/t/p/w500/' + poster_path
    return full_path

# Function to recommend movies
def recommender(movie):
    movie_index = df[df['title'] == movie].index[0]
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:21]
    recommended_titles = []
    recommended_posters = []
    for i in movies_list:
        movie_id = df.iloc[i[0]]['movie_id']
        recommended_titles.append(df.iloc[i[0]]['title'])
        recommended_posters.append(fetch_poster(movie_id))
    return recommended_titles, recommended_posters

# Configure Streamlit
st.set_page_config(layout="wide")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title('CINEPHILE ENGINE')
selected_movie = st.selectbox('Type a Movie', options=titles)

# Display recommended movies and posters when the button is clicked
if st.button('Recommend'):
    recommended_movie_names, recommended_movie_posters = recommender(selected_movie)

    # Use containers to align elements
    rows = 4  # Rows to hold 20 movies
    cols_per_row = 5  # 5 movies per row

    for row in range(rows):
        with st.container():
            cols = st.columns(cols_per_row)
            start_idx = row * cols_per_row
            end_idx = start_idx + cols_per_row
            for i, col in zip(range(start_idx, end_idx), cols):
                if i < len(recommended_movie_names):
                    col.text(recommended_movie_names[i])
                    col.image(recommended_movie_posters[i])
