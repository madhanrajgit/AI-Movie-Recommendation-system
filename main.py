import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# TMDb API Key (Ensure it's secured properly!)
API_KEY = "887f725faa2dadb468b5baef8c697023"

# Load dataset safely
def load_data():
    try:
        df = pd.read_csv("merged_movies.csv")
        if df.empty:
            raise pd.errors.EmptyDataError  # If no data is present, raise an error
        df.columns = df.columns.str.strip()
        df.fillna("Information not available", inplace=True)
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("⚠️ Error: Dataset `merged_movies.csv` is missing or empty.")
        return pd.DataFrame()  # Initialize an empty DataFrame

df = load_data()

# TF-IDF vectorization for recommendations (only if data exists)
if not df.empty:
    tfidf = TfidfVectorizer(stop_words="english")
    vector = tfidf.fit_transform(df["overview"])
    similarity = cosine_similarity(vector)

# Function to fetch movie details from TMDb
def get_movie_info(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
    response = requests.get(url).json()
    if response["results"]:
        result = response["results"][0]
        return {
            "poster_url": f"https://image.tmdb.org/t/p/w500{result['poster_path']}" if result.get("poster_path") else None,
            "rating": result.get("vote_average", "N/A"),
            "vote_count": result.get("vote_count", "N/A"),
            "id": result["id"]
        }
    return {"poster_url": None, "rating": "N/A", "vote_count": "N/A", "id": None}

# Function to generate recommendations
def recommend(movie_title):
    movie_title_lower = movie_title.lower()
    movie_list = df["title"].str.lower().tolist()
    best_match = process.extractOne(movie_title_lower, movie_list)

    if not best_match or best_match[1] < 80:
        return None

    matched_title = best_match[0]
    idx = movie_list.index(matched_title)
    recommended_movies = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)[1:6]
    recommended_titles = [df.loc[i[0], "title"] for i in recommended_movies]

    return {
        "searched_movie": matched_title,
        "recommended_movies": recommended_titles
    }
