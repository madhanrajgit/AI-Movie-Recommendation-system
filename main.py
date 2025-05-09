import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("merged_movies.csv")
df.columns = df.columns.str.strip()

# TF-IDF for recommendations
tfidf = TfidfVectorizer(stop_words="english")
vector = tfidf.fit_transform(df["overview"].fillna(""))
similarity = cosine_similarity(vector)

# Recommendation function
def recommend(movie_title):
    movie_title_lower = movie_title.lower()
    movie_list = df["title"].str.lower().tolist()

    if movie_title_lower in movie_list:
        idx = movie_list.index(movie_title_lower)

        # Show the searched movie's details
        searched_movie_title = df.loc[idx, "title"]
        searched_movie_overview = df.loc[idx, "overview"]
        print(f"✅ Your searched movie: {searched_movie_title}")
        print(f"📖 Overview: {searched_movie_overview}\n")
        
        recommended_movies = sorted(
            list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True
        )[1:6]

        for i in recommended_movies:
            print(f"👉 {df.loc[i[0], 'title']}")
            print(f"📖 {df.loc[i[0], 'overview']}\n")
    else:
        print(f"❌ Movie '{movie_title}' not found. Showing top popular movies!\n")
        top_popular = df.sort_values("popularity", ascending=False).head(5)

        for _, row in top_popular.iterrows():
            print(f"👉 {row['title']}")
            print(f"📖 {row['overview']}\n")

# Example usage
if __name__ == "__main__":
    recommend("Baahubali: The Beginning")
    recommend("Unknown Movie")
