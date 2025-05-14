import streamlit as st
import pandas as pd
import requests
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process, fuzz
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import pyrebase
import bcrypt
import gdown
import json

# --- Firebase Configuration ---
firebase_config = {
    "apiKey": "AIzaSyALfZTY7lHMMtk1gGH81mvjUONznrUkXbE",
    "authDomain": "movie-recommender-2025.firebaseapp.com",
    "databaseURL": "https://movie-recommender-2025.firebaseio.com",
    "projectId": "movie-recommender-2025",
    "storageBucket": "movie-recommender-2025.firebasestorage.app",
    "messagingSenderId": "290219049070",
    "appId": "1:290219049070:web:1b55bf641cf9f98d16e67f"
}

try:
    firebase = pyrebase.initialize_app(firebase_config)
    auth = firebase.auth()
    db = firebase.database()
except Exception as e:
    st.error(f"Firebase initialization failed: {e}")
    st.stop()

# --- TMDb API Configuration ---
try:
    API_KEY = st.secrets["API_KEY"]
except KeyError:
    API_KEY = os.getenv("TMDB_API_KEY")
    if not API_KEY:
        st.error("TMDb API key not found.")
        st.stop()

# --- Parse Genres ---
def parse_genres(genre_str):
    if not genre_str or pd.isna(genre_str):
        return ""
    try:
        genres = json.loads(genre_str.replace("'", "\""))
        return ",".join([g["name"] for g in genres])
    except:
        return genre_str if isinstance(genre_str, str) else ""

# --- Data Loading ---
@st.cache_data
def load_data(tmdb_file="data/tmdb_movies.csv"):
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        gdown.download("https://drive.google.com/uc?id=your_file_id", os.path.join(data_dir, "tmdb_movies.csv"), quiet=False)
    
    try:
        df = pd.read_csv(tmdb_file, usecols=[
            "id", "title", "overview", "genres", "popularity",
            "vote_average", "vote_count", "poster_path", "release_date"
        ])
    except Exception as e:
        st.error(f"Error loading TMDb dataset: {e}")
        st.stop()
    
    df["title"] = df["title"].fillna("Unknown Title")
    df["overview"] = df["overview"].fillna("Overview not available")
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    df["genres"] = df["genres"].apply(parse_genres)
    df["vote_average"] = df["vote_average"].fillna("N/A")
    df["vote_count"] = df["vote_count"].fillna("N/A")
    df["poster_path"] = df["poster_path"].fillna("")
    df["release_date"] = df["release_date"].fillna("N/A")
    
    return df, [tmdb_file]

# --- Title Index ---
@st.cache_data
def build_title_index(files):
    titles = set()
    for file in files:
        try:
            df = pd.read_csv(file, usecols=["title"])
            titles.update(df["title"].str.lower().tolist())
        except Exception as e:
            st.warning(f"Error building title index for {file}: {e}")
            continue
    return list(titles)

# --- TF-IDF Similarity ---
@st.cache_resource
def compute_similarity(df, file_index=0):
    tfidf = TfidfVectorizer(stop_words="english")
    vector = tfidf.fit_transform(df["overview"])
    similarity = cosine_similarity(vector)
    with open(f"similarity_{file_index}.pkl", "wb") as f:
        pickle.dump(similarity, f)
    return similarity

# --- TMDb API Functions ---
@st.cache_data
def get_movie_info(movie_id):
    try:
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
        credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}"
        
        details_response = requests.get(details_url)
        details_response.raise_for_status()
        data = details_response.json()
        
        credits_response = requests.get(credits_url)
        credits_response.raise_for_status()
        credits = credits_response.json()
        
        director = next((crew["name"] for crew in credits.get("crew", []) if crew["job"] == "Director"), "N/A")
        cast = [actor["name"] for actor in credits.get("cast", [])[:3]]
        return {
            "poster_url": f"https://image.tmdb.org/t/p/w500{data['poster_path']}" if data.get("poster_path") else None,
            "rating": data.get("vote_average", "N/A"),
            "vote_count": data.get("vote_count", "N/A"),
            "genres": [genre["id"] for genre in data.get("genres", [])],
            "director": director,
            "cast": cast,
            "movie_id": movie_id
        }
    except requests.RequestException as e:
        st.error(f"Error fetching movie info for ID {movie_id}: {e}")
        return {"poster_url": None, "rating": "N/A", "vote_count": "N/A", "genres": [], "director": "N/A", "cast": [], "movie_id": movie_id}

@st.cache_data
def get_top_rated_by_genre(genre_id):
    try:
        url = f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&sort_by=vote_average.desc&with_genres={genre_id}&vote_count.gte=100"
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("results", [])[:5]
    except requests.RequestException as e:
        st.error(f"Error fetching top-rated movies for genre ID {genre_id}: {e}")
        return []

@st.cache_data
def get_movies_by_crew(crew_name, exclude_movie_id):
    try:
        url = f"https://api.themoviedb.org/3/search/person?api_key={API_KEY}&query={crew_name}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data["results"]:
            person_id = data["results"][0]["id"]
            credits_url = f"https://api.themoviedb.org/3/person/{person_id}/movie_credits?api_key={API_KEY}"
            credits_response = requests.get(credits_url)
            credits_response.raise_for_status()
            movies = credits_response.json().get("cast", []) + credits_response.json().get("crew", [])
            return [movie for movie in movies if movie["id"] != exclude_movie_id][:5]
    except requests.RequestException as e:
        st.error(f"Error fetching movies for crew '{crew_name}': {e}")
        return []

@st.cache_data
def fetch_new_movies():
    try:
        url = f"https://api.themoviedb.org/3/movie/now_playing?api_key={API_KEY}&language=en-US&page=1"
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("results", [])[:10]
    except requests.RequestException as e:
        st.error(f"Error fetching new movies: {e}")
        return []

# --- User Management ---
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def register_user(email, password):
    try:
        user = auth.create_user_with_email_and_password(email, password)
        hashed_password = hash_password(password)
        db.child("users").child(user["localId"]).set({"email": email, "password": hashed_password})
        return user
    except Exception as e:
        st.error(f"Registration failed: {e}")
        return None

def login_user(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        return user
    except Exception as e:
        st.error(f"Login failed: {e}")
        return None

# --- Personalization ---
def save_user_rating(user_id, movie_title, rating):
    try:
        db.child("ratings").child(user_id).child(movie_title).set({"rating": rating})
    except Exception as e:
        st.error(f"Error saving rating: {e}")

def get_user_ratings(user_id):
    try:
        ratings = db.child("ratings").child(user_id).get().val()
        if ratings:
            return [(movie, data["rating"]) for movie, data in ratings.items()]
        return []
    except Exception as e:
        st.error(f"Error fetching ratings: {e}")
        return []

def collaborative_filtering(user_id, df):
    user_ratings = get_user_ratings(user_id)
    if not user_ratings:
        return []
    
    ratings_data = [(user_id, movie, rating) for movie, rating in user_ratings]
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(pd.DataFrame(ratings_data, columns=["user_id", "movie_title", "rating"]), reader)
    trainset, _ = train_test_split(data, test_size=0.2)
    
    algo = SVD()
    algo.fit(trainset)
    
    predictions = []
    for movie in df["title"].unique():
        pred = algo.predict(user_id, movie)
        predictions.append((movie, pred.est))
    
    return sorted(predictions, key=lambda x: x[1], reverse=True)[:5]

# --- Recommendation Logic ---
def recommend(movie_title, df, similarity, files, user_id=None):
    results = {"searched_movie": None, "same_genre": [], "same_crew": [], "top_rated_genre": [], "personalized": []}
    
    if user_id:
        results["personalized"] = collaborative_filtering(user_id, df)
    
    found = False
    for file in files:
        try:
            temp_df = pd.read_csv(file, usecols=[
                "id", "title", "overview", "genres", "popularity",
                "vote_average", "vote_count", "poster_path", "release_date"
            ])
            temp_df["title_lower"] = temp_df["title"].str.lower()
            if movie_title.lower() in temp_df["title_lower"].values:
                idx = temp_df[temp_df["title_lower"] == movie_title.lower()].index[0]
                searched_movie_id = temp_df.loc[idx, "id"]
                searched_movie_title = temp_df.loc[idx, "title"]
                searched_movie_overview = temp_df.loc[idx, "overview"]
                searched_movie_genres = temp_df.loc[idx, "genres"].split(",") if temp_df.loc[idx, "genres"] else []
                movie_info = get_movie_info(searched_movie_id)

                results["searched_movie"] = {
                    "id": searched_movie_id,
                    "title": searched_movie_title,
                    "overview": searched_movie_overview,
                    "genres": searched_movie_genres,
                    "director": movie_info["director"],
                    "cast": movie_info["cast"],
                    "movie_info": movie_info,
                    "vote_average": temp_df.loc[idx, "vote_average"],
                    "vote_count": temp_df.loc[idx, "vote_count"],
                    "poster_path": f"https://image.tmdb.org/t/p/w500{temp_df.loc[idx, 'poster_path']}" if temp_df.loc[idx, "poster_path"] else None,
                    "release_date": temp_df.loc[idx, "release_date"]
                }
                found = True
                break
        except Exception as e:
            st.warning(f"Error processing file {file}: {e}")
            continue
    
    if not found:
        movie_list = build_title_index(files)
        match = process.extractOne(movie_title.lower(), movie_list, scorer=fuzz.token_sort_ratio)
        if match and match[1] > 80:
            for file in files:
                try:
                    temp_df = pd.read_csv(file, usecols=[
                        "id", "title", "overview", "genres", "popularity",
                        "vote_average", "vote_count", "poster_path", "release_date"
                    ])
                    temp_df["title_lower"] = temp_df["title"].str.lower()
                    if match[0] in temp_df["title_lower"].values:
                        idx = temp_df[temp_df["title_lower"] == match[0]].index[0]
                        searched_movie_id = temp_df.loc[idx, "id"]
                        searched_movie_title = temp_df.loc[idx, "title"]
                        searched_movie_overview = temp_df.loc[idx, "overview"]
                        searched_movie_genres = temp_df.loc[idx, "genres"].split(",") if temp_df.loc[idx, "genres"] else []
                        movie_info = get_movie_info(searched_movie_id)

                        results["searched_movie"] = {
                            "id": searched_movie_id,
                            "title": searched_movie_title,
                            "overview": searched_movie_overview,
                            "genres": searched_movie_genres,
                            "director": movie_info["director"],
                            "cast": movie_info["cast"],
                            "movie_info": movie_info,
                            "vote_average": temp_df.loc[idx, "vote_average"],
                            "vote_count": temp_df.loc[idx, "vote_count"],
                            "poster_path": f"https://image.tmdb.org/t/p/w500{temp_df.loc[idx, 'poster_path']}" if temp_df.loc[idx, "poster_path"] else None,
                            "release_date": temp_df.loc[idx, "release_date"]
                        }
                        found = True
                        break
                except Exception as e:
                    st.warning(f"Error processing file {file}: {e}")
                    continue
    
    if found:
        if results["searched_movie"]["genres"]:
            genre_matches = []
            for file in files:
                try:
                    temp_df = pd.read_csv(file, usecols=["title", "overview", "genres"])
                    temp_df["genres"] = temp_df["genres"].apply(parse_genres)
                    matches = temp_df[temp_df["genres"].str.contains("|".join(results["searched_movie"]["genres"]), case=False, na=False)]
                    matches = matches[matches["title"] != results["searched_movie"]["title"]]
                    genre_matches.append(matches.head(5 - len(genre_matches)))
                    if len(genre_matches) >= 5:
                        break
                except Exception as e:
                    st.warning(f"Error processing file {file} for genre matches: {e}")
                    continue
            if genre_matches:
                genre_df = pd.concat(genre_matches, ignore_index=True).head(5)
                results["same_genre"] = [
                    {"title": row["title"], "overview": row["overview"]}
                    for _, row in genre_df.iterrows()
                ]

        if results["searched_movie"]["director"] != "N/A":
            crew_movies = get_movies_by_crew(results["searched_movie"]["director"], results["searched_movie"]["movie_info"]["movie_id"])
            results["same_crew"] = [
                {"title": movie["title"], "overview": movie.get("overview", "Overview not available")}
                for movie in crew_movies
            ]

        if results["searched_movie"]["movie_info"]["genres"]:
            primary_genre_id = results["searched_movie"]["movie_info"]["genres"][0]
            top_rated = get_top_rated_by_genre(primary_genre_id)
            results["top_rated_genre"] = [
                {"title": movie["title"], "overview": movie["overview"], "rating": movie["vote_average"], "vote_count": movie["vote_count"]}
                for movie in top_rated
            ]
    else:
        st.subheader(f"❌ Movie '{movie_title}' not found. Showing top popular movies!")
        top_popular = df.sort_values("popularity", ascending=False).head(5)
        results["same_genre"] = [
            {"title": row["title"], "overview": row["overview"]}
            for _, row in top_popular.iterrows()
        ]

    return results

# --- Autocomplete Search ---
@st.cache_data
def get_search_suggestions(query, movie_list, limit=5):
    if not query.strip():
        return []
    matches = process.extract(query.lower(), movie_list, scorer=fuzz.partial_ratio, limit=limit)
    return [match[0] for match in matches if match[1] > 70]
