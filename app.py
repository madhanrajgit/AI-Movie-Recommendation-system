import streamlit as st
import main  # Import backend logic from main.py
from streamlit_autorefresh import st_autorefresh

# UI Styling
st.markdown("""
    <style>
        body { background-color: #121212; color: white; font-family: Arial, sans-serif; }
        .title { font-size: 24px; font-weight: bold; color: gold; text-align: center; }
        .rating { font-size: 18px; color: lightgreen; }
        .overview { font-size: 14px; color: white; }
        .movie-container { border: 1px solid #444; padding: 10px; border-radius: 10px; background-color: #222; text-align: center; }
        .button { background-color: gold; color: black; font-size: 16px; padding: 5px; border-radius: 8px; }
        .ad-image { width: 30%; margin: auto; display: block; }
    </style>
""", unsafe_allow_html=True)

# UI Layout
st.title("🎬 IMDb-Style AI Movie Recommender")
col1, col2 = st.columns([4, 1])

with col1:
    movie_input = st.text_input("🔍 Search for a movie:", "")

with col2:
    if st.button("🔍 Search") and movie_input:
        movie_result = main.recommend(movie_input)
        if movie_result:
            st.session_state.selected_movie = movie_result["searched_movie"]
            st.session_state.recommended_movies = movie_result["recommended_movies"]
        else:
            st.warning("⚠️ Movie not found. Try another title!")

# Display Movie Details if Selected
if "selected_movie" in st.session_state:
    movie_info = main.get_movie_info(st.session_state.selected_movie)
    
    st.subheader(f"🎬 {st.session_state.selected_movie}")

    if movie_info["poster_url"]:
        st.image(movie_info["poster_url"], caption=st.session_state.selected_movie, use_container_width=True)

    st.markdown(f"⭐ **Rating:** {movie_info['rating']} / 10 ({movie_info['vote_count']} votes)")

    # Show Recommendations
    st.subheader("🔹 Recommended Movies")
    for movie in st.session_state.recommended_movies:
        st.write(f"🎬 [{movie}](?selected_movie={movie})")

# Auto-Sliding Featured Movies
if not movie_input or st.button("🏠 Back to Home"):
    trending_movies = [main.get_movie_info(title) for title in main.df.sample(4)["title"].tolist()]
    for movie in trending_movies:
        st.image(movie["poster_url"], use_container_width=True, caption=f"🔥 Featured: {movie['id']}")
