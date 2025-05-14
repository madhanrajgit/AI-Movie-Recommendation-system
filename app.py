import streamlit as st
from main import (load_data, build_title_index, compute_similarity, recommend,
                 get_search_suggestions, fetch_new_movies, get_top_rated_by_genre,
                 login_user, register_user, save_user_rating, get_user_ratings)

def main():
    st.title("🎬 AI Movie Recommender")

    # Initialize session state
    if "user" not in st.session_state:
        st.session_state.user = None
    if "page" not in st.session_state:
        st.session_state.page = "login"

    # Load data
    df, files = load_data()
    try:
        with open("similarity_0.pkl", "rb") as f:
            similarity = pickle.load(f)
    except FileNotFoundError:
        similarity = compute_similarity(df, file_index=0)
    
    movie_list = build_title_index(files)

    # User Authentication
    if st.session_state.user is None:
        if st.session_state.page == "login":
            st.subheader("Login")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                user = login_user(email, password)
                if user:
                    st.session_state.user = user
                    st.success("Logged in successfully!")
                    st.rerun()
            if st.button("Go to Register"):
                st.session_state.page = "register"
                st.rerun()
        
        elif st.session_state.page == "register":
            st.subheader("Register")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Register"):
                user = register_user(email, password)
                if user:
                    st.session_state.user = user
                    st.success("Registered and logged in successfully!")
                    st.rerun()
            if st.button("Go to Login"):
                st.session_state.page = "login"
                st.rerun()
    else:
        st.subheader(f"Welcome, {st.session_state.user['email']}")
        if st.button("Logout"):
            st.session_state.user = None
            st.session_state.page = "login"
            st.rerun()

        # Search and Recommendations
        st.subheader("Search for a Movie")
        query = st.text_input("Enter a movie name:", "", key="search_input")
        suggestions = get_search_suggestions(query, movie_list)
        
        selected_movie = None
        if suggestions:
            suggestion_options = [""] + suggestions
            selected_movie = st.selectbox("Suggestions:", suggestion_options, index=0, key="suggestions")
        
        movie_input = selected_movie if selected_movie else query

        if st.button("Recommend"):
            if movie_input.strip():
                results = recommend(movie_input, df, similarity, files, user_id=st.session_state.user["localId"])
                
                # Searched Movie
                if results["searched_movie"]:
                    searched = results["searched_movie"]
                    st.subheader(f"✅ Your Searched Movie: {searched['title']}")
                    st.markdown(f"📅 **Release Date:** {searched['release_date']}")
                    st.markdown(f"📖 **Overview:** {searched['overview']}")
                    st.markdown(f"⭐ **Rating:** {searched['vote_average']} / 10 ({searched['vote_count']} votes)")
                    st.markdown(f"🎭 **Genres:** {', '.join(searched['genres']) if searched['genres'] else 'N/A'}")
                    st.markdown(f"🎬 **Director:** {searched['director']}")
                    st.markdown(f"🌟 **Cast:** {', '.join(searched['cast']) if searched['cast'] else 'N/A'}")
                    if searched["poster_path"]:
                        st.image(searched["poster_path"], caption=searched["title"], width=200)
                    else:
                        st.image("https://via.placeholder.com/500x750.png?text=No+Poster+Available", caption="No Poster Available", width=200)
                    
                    # Rate Movie
                    rating = st.slider("Rate this movie (1-10):", 1, 10, 5)
                    if st.button("Submit Rating"):
                        save_user_rating(st.session_state.user["localId"], searched["title"], rating)
                        st.success("Rating saved!")

                # Personalized Recommendations
                if results["personalized"]:
                    st.subheader("🎯 Personalized Recommendations")
                    cols = st.columns(3)
                    for i, (movie, rating) in enumerate(results["personalized"]):
                        with cols[i % 3]:
                            st.write(f"**👉 {movie}**")
                            st.markdown(f"⭐ **Predicted Rating:** {rating:.1f} / 10")
                            temp_df = df[df["title"].str.lower() == movie.lower()]
                            poster = f"https://image.tmdb.org/t/p/w500{temp_df['poster_path'].iloc[0]}" if not temp_df.empty and temp_df["poster_path"].iloc[0] else None
                            if poster:
                                st.image(poster, caption=movie, width=150)
                            else:
                                st.image("https://via.placeholder.com/500x750.png?text=No+Poster+Available", caption="No Poster Available", width=150)

                # Same Genre
                if results["same_genre"]:
                    st.subheader("🎥 Movies in the Same Genre")
                    cols = st.columns(3)
                    for i, movie in enumerate(results["same_genre"]):
                        with cols[i % 3]:
                            st.write(f"**👉 {movie['title']}**")
                            st.markdown(f"📖 **Overview:** {movie['overview']}")
                            temp_df = df[df["title"].str.lower() == movie["title"].lower()]
                            rating = temp_df["vote_average"].iloc[0] if not temp_df.empty else "N/A"
                            vote_count = temp_df["vote_count"].iloc[0] if not temp_df.empty else "N/A"
                            poster = f"https://image.tmdb.org/t/p/w500{temp_df['poster_path'].iloc[0]}" if not temp_df.empty and temp_df["poster_path"].iloc[0] else None
                            st.markdown(f"⭐ **Rating:** {rating} / 10 ({vote_count} votes)")
                            if poster:
                                st.image(poster, caption=movie["title"], width=150)
                            else:
                                st.image("https://via.placeholder.com/500x750.png?text=No+Poster+Available", caption="No Poster Available", width=150)

                # Same Crew
                if results["same_crew"]:
                    st.subheader("🎬 Movies by the Same Director")
                    cols = st.columns(3)
                    for i, movie in enumerate(results["same_crew"]):
                        with cols[i % 3]:
                            st.write(f"**👉 {movie['title']}**")
                            st.markdown(f"📖 **Overview:** {movie['overview']}")
                            temp_df = df[df["title"].str.lower() == movie["title"].lower()]
                            rating = temp_df["vote_average"].iloc[0] if not temp_df.empty else "N/A"
                            vote_count = temp_df["vote_count"].iloc[0] if not temp_df.empty else "N/A"
                            poster = f"https://image.tmdb.org/t/p/w500{temp_df['poster_path'].iloc[0]}" if not temp_df.empty and temp_df["poster_path"].iloc[0] else None
                            st.markdown(f"⭐ **Rating:** {rating} / 10 ({vote_count} votes)")
                            if poster:
                                st.image(poster, caption=movie["title"], width=150)
                            else:
                                st.image("https://via.placeholder.com/500x750.png?text=No+Poster+Available", caption="No Poster Available", width=150)

                # Top-Rated by Genre
                if results["top_rated_genre"]:
                    st.subheader("🏆 Top-Rated Movies in the Same Genre")
                    cols = st.columns(3)
                    for i, movie in enumerate(results["top_rated_genre"]):
                        with cols[i % 3]:
                            st.write(f"**👉 {movie['title']}**")
                            st.markdown(f"📖 **Overview:** {movie['overview']}")
                            st.markdown(f"⭐ **Rating:** {movie['rating']} / 10 ({movie['vote_count']} votes)")
                            poster_url = get_movie_info(movie["title"])["poster_url"]
                            if poster_url:
                                st.image(poster_url, caption=movie["title"], width=150)
                            else:
                                st.image("https://via.placeholder.com/500x750.png?text=No+Poster+Available", caption="No Poster Available", width=150)
            else:
                st.warning("Please enter a movie title!")

        # New Movies from TMDb
        if st.button("Show New Releases"):
            new_movies = fetch_new_movies()
            if new_movies:
                st.subheader("🎬 New Releases")
                cols = st.columns(3)
                for i, movie in enumerate(new_movies):
                    with cols[i % 3]:
                        st.write(f"**👉 {movie['title']}**")
                        st.markdown(f"📖 **Overview:** {movie['overview']}")
                        st.markdown(f"⭐ **Rating:** {movie['vote_average']} / 10 ({movie['vote_count']} votes)")
                        poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None
                        if poster_url:
                            st.image(poster_url, caption=movie["title"], width=150)
                        else:
                            st.image("https://via.placeholder.com/500x750.png?text=No+Poster+Available", caption="No Poster Available", width=150)

        # General Top-Rated Movies
        if st.button("Top Rated Movies"):
            top_movies = get_top_rated_by_genre("")
            if top_movies:
                st.subheader("🎯 Top Rated Movies (All Genres)")
                cols = st.columns(3)
                for i, movie in enumerate(top_movies):
                    with cols[i % 3]:
                        st.write(f"**👉 {movie['title']}**")
                        st.markdown(f"⭐ **Rating:** {movie['vote_average']} / 10 ({movie['vote_count']} votes)")
                        st.markdown(f"📖 **Overview:** {movie['overview']}")
                        poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None
                        if poster_url:
                            st.image(poster_url, caption=movie["title"], width=150)
                        else:
                            st.image("https://via.placeholder.com/500x750.png?text=No+Poster+Available", caption="No Poster Available", width=150)

if __name__ == "__main__":
    main()
