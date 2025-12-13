import streamlit as st
import pandas as pd
import pickle
import requests
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F3EEE6;
        color: #2E2E2E;
        font-family: "Segoe UI", sans-serif;
    }

    .app-title {
        text-align: center;
        font-size: 52px;
        font-weight: 800;
        color: #7A4A2E;
        margin-bottom: 30px;
        letter-spacing: 1px;
    }

    .stSelectbox label {
        color: #2E2E2E;
        font-size: 18px;
        font-weight: 600;
    }

    .stButton>button {
        background-color: #7A4A2E;
        color: white;
        border-radius: 8px;
        padding: 10px 22px;
        font-size: 17px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #5C3A21;
        transform: scale(1.05);
    }

    /* Movie posters */ 
    img { 
    border-radius: 14px; 
    box-shadow: 0 8px 18px rgba(92,58,33,0.25); 
    transition: transform 0.3s ease; 
    } 
    
    img:hover { 
    transform: scale(1.08); 
    } 
    
    /* Movie title under poster */ 
    .movie-name { 
    text-align: center; 
    font-size: 15px; 
    margin-top: 10px; 
    font-weight: 600; 
    color: #2E2E2E; 
    }

    </style>
    """,
    unsafe_allow_html=True
)

movies_dict = pickle.load(open('movies.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

@st.cache_resource
def compute_similarity(df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    return cosine_similarity(vectors)

similarity = compute_similarity(movies)

API_KEY = os.getenv("OMDB_API_KEY")

def fetch_poster(movie_title):
    response = requests.get(
        f"https://www.omdbapi.com/?t={movie_title}&apikey={API_KEY}"
    )
    data = response.json()
    return data.get('Poster')

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_movie_posters = []

    for i in movie_list:
        title = movies.iloc[i[0]].title
        recommended_movies.append(title)
        recommended_movie_posters.append(fetch_poster(title))

    return recommended_movies, recommended_movie_posters

st.markdown("<div class='app-title'>🎬 YehDekho</div>", unsafe_allow_html=True)

selected_movie_name = st.selectbox(
    'Select a movie you like:',
    movies['title'].values
)

if st.button('Recommend'):
    names, posters = recommend(selected_movie_name)

    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            st.markdown(
                f"""
                <div class="movie-card">
                    <img src="{posters[idx]}"/>
                    <div class="movie-name">{names[idx]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
