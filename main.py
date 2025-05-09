import pandas as pd

# Load the dataset
df = pd.read_csv("merged_movies.csv")

# Ensure column names are clean
df.columns = df.columns.str.strip()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Convert movie descriptions into TF-IDF vectors
tfidf = TfidfVectorizer(stop_words='english')
vector = tfidf.fit_transform(df['overview'].fillna(''))  # Ensure no missing values

# Compute cosine similarity matrix
similarity = cosine_similarity(vector)

# If a movie is found, show recommendations
def recommend(movie_title):
    movie_title_lower = movie_title.lower()
    movie_list = df['title'].str.lower().tolist()

    print("🔍 Searching for your movie...")

def recommend(movie_title):
    movie_title_lower = movie_title.lower()
    movie_list = df['title'].str.lower().tolist()

    print("🔍 Searching for your movie...")

    if movie_title_lower in movie_list:
        idx = movie_list.index(movie_title_lower)
        print(f"\n✅ Your searched movie **'{df.loc[idx, 'title']}'** was found in the dataset.")
        print(f"📖 Overview: {df.loc[idx, 'overview']}\n")
        print("🎬 Recommended similar movies:\n")

        distances = list(enumerate(similarity[idx]))
        distances = sorted(distances, key=lambda x: x[1], reverse=True)

        for i in distances[1:6]:
            print(f"👉 {df.loc[i[0], 'title']}\n📖 {df.loc[i[0], 'overview']}\n")
    else:
        print(f"\n❌ Your searched movie **'{movie_title}'** was NOT found in the dataset.")
        print("\n🎯 Showing Top 5 Most Popular Movies instead:\n")

        top_popular = df.sort_values('popularity', ascending=False).head(5)
        for i, row in top_popular.iterrows():
            print(f"👉 {row['title']}\n📖 {row['overview']}\n")

# Test the function
if __name__ == "__main__":
    recommend("Baahubali: The Beginning")  # ✅ Works if found
    recommend("Unknown Movie")  # ❌ Shows most popular movies instead
