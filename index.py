import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
movies = pd.DataFrame({
    'title': [
        'Inception', 'The Matrix', 'Interstellar', 'The Godfather', 'The Dark Knight',
        'Pulp Fiction', 'Fight Club', 'Forrest Gump', 'Gladiator', 'Titanic'
    ],
    'genre': [
        'Sci-Fi Thriller', 'Sci-Fi Action', 'Sci-Fi Drama', 'Crime Drama', 'Action Crime',
        'Crime Drama', 'Drama Thriller', 'Romantic Drama', 'Historical Action', 'Romantic Drama'
    ]
})

# AI Matching System - Content Based Filtering
def recommend_movies(user_genre_input, top_n=5):
    tfidf = TfidfVectorizer(stop_words='english')
    genre_matrix = tfidf.fit_transform(movies['genre'])

    user_vec = tfidf.transform([user_genre_input])
    cosine_sim = cosine_similarity(user_vec, genre_matrix).flatten()

    recommended_indices = cosine_sim.argsort()[-top_n:][::-1]
    return movies.iloc[recommended_indices]

# Streamlit Frontend
st.title("AI-Driven Movie Matchmaker")
st.write("Get personalized movie recommendations based on your preferences!")

user_input = st.text_input("Describe your ideal movie (e.g., 'sci-fi thriller', 'romantic drama'):")

if user_input:
    st.subheader("Recommended Movies for You:")
    recommendations = recommend_movies(user_input)
    for index, row in recommendations.iterrows():
        st.write(f"**{row['title']}** - *{row['genre']}*")
