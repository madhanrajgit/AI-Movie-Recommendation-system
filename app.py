import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("merged_movies.csv")
df.columns = df.columns.str.strip()

# Fill NaN values in overview column
df["overview"].fillna("Overview not available", inplace=True)

# TF-IDF for recommendations
tfidf = TfidfVectorizer(stop_words="english")
vector = tfidf.fit_transform(df["overview"].fillna(""))
similarity = cosine_similarity(vector)

# Function to get movie posters from IMDb
def get_poster_url(movie_title):
    try:
        search_url = f"https://www.imdb.com/find?q={movie_title.replace(' ', '+')}&s=tt"
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, "html.parser")
        result = soup.find("td", class_="result_text")

        if result:
            movie_url = "https://www.imdb.com" + result.a["href"]
            response = requests.get(movie_url)
            soup = BeautifulSoup(response.text, "html.parser")
            poster_element = soup.find("div", class_="poster")

            if poster_element and poster_element.img:
                return poster_element.img["src"]

        return None
    except Exception as e:
        return None

# Recommendation function
def recommend(movie_title):
    movie_title_lower = movie_title.lower()
    movie_list = df["title"].str.lower().tolist()
    results = []  # Store recommended movies

    if movie_title_lower in movie_list:
        idx = movie_list.index(movie_title_lower)

        # Display searched movie's overview and poster
        searched_movie_title = df.loc[idx, "title"]
        searched_movie_overview = df.loc[idx, "overview"]
        poster_url = get_poster_url(searched_movie_title)

        st.subheader(f"✅ Your searched movie: {searched_movie_title}")
        st.markdown(f"📖 **Overview:** {searched_movie_overview}")

        if poster_url:
            st.image(poster_url, caption=searched_movie_title)
        else:
            st.image("default_poster.jpg", caption="No Poster Available")

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
st.title("🎬 AI Movie Recommender")

movie_input = st.text_input("Enter a movie name:", "")

if st.button("Recommend"):
    if movie_input:
        results = recommend(movie_input)

        for movie in results:
            st.write(f"**👉 {movie['title']}**")
            st.markdown(f"📖 **Overview:** {movie['overview']}")

            # Display poster for recommended movies
            poster_url = get_poster_url(movie["title"])
            if poster_url:
                st.image(poster_url, caption=movie["title"])
            else:
                st.image("default_poster.jpg", caption="No Poster Available")
