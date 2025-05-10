import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TMDb API Key (Replace with your actual API key)
API_KEY = "887f725faa2dadb468b5baef8c697023"

# Load dataset
df = pd.read_csv("merged_movies.csv")
df.columns = df.columns.str.strip()
df["overview"].fillna("Overview not available", inplace=True)

# TF-IDF for recommendations
tfidf = TfidfVectorizer(stop_words="english")
vector = tfidf.fit_transform(df["overview"])
similarity = cosine_similarity(vector)

# TMDb: Get movie poster
def get_movie_poster(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
    response = requests.get(url)
    data = response.json()
    if data["results"]:
        poster_path = data["results"][0].get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return "https://via.placeholder.com/500x750.png?text=No+Poster+Available"

# TMDb: Get movie rating
def get_movie_rating(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
    response = requests.get(url)
    data = response.json()
    if data["results"]:
        movie_data = data["results"][0]
        rating = movie_data.get("vote_average", "N/A")
        vote_count = movie_data.get("vote_count", "N/A")
        return rating, vote_count
    return "N/A", "N/A"

# Recommendation system
def recommend(movie_title):
    movie_title_lower = movie_title.lower()
    movie_list = df["title"].str.lower().tolist()
    results = []

    if movie_title_lower in movie_list:
        idx = movie_list.index(movie_title_lower)
        searched_movie_title = df.loc[idx, "title"]
        searched_movie_overview = df.loc[idx, "overview"]
        poster_url = get_movie_poster(searched_movie_title)
        rating, vote_count = get_movie_rating(searched_movie_title)

        # Display main movie
        st.subheader(f"✅ Your searched movie: {searched_movie_title}")
        st.markdown(f"⭐ **Rating:** {rating} ({vote_count} votes)")
        st.markdown(f"📖 **Overview:** {searched_movie_overview}")
        st.image(poster_url, width=300)

        # Top 5 similar movies
        recommended_movies = sorted(
            list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True
        )[1:6]

        results = [
            {"title": df.loc[i[0], "title"], "overview": df.loc[i[0], "overview"]}
            for i in recommended_movies
        ]
    else:
        st.subheader(f"❌ Movie '{movie_title}' not found. Showing top popular movies!")
        top_popular = df.sort_values("popularity", ascending=False).head(5)
        results = [
            {"title": row["title"], "overview": row["overview"]}
            for _, row in top_popular.iterrows()
        ]

    return results

# Streamlit UI
st.set_page_config(page_title="AI Movie Recommender", layout="wide")
st.title("🎬 AI Movie Recommender System")
st.markdown("Enter a movie name to get recommendations based on its **overview** and get **ratings** from TMDb.")

movie_input = st.text_input("🔎 Enter a movie name:")

if st.button("📽️ Recommend"):
    if movie_input.strip() == "":
        st.warning("Please enter a movie name!")
    else:
        results = recommend(movie_input)
        for movie in results:
            st.write(f"---\n### 👉 {movie['title']}")
            st.markdown(f"📖 **Overview:** {movie['overview']}")
            st.image(get_movie_poster(movie['title']), width=250)
