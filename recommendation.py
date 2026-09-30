import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies = pd.DataFrame({
    "title": [
        "Interstellar", "Inception", "The Martian", "Avengers: Endgame",
        "The Dark Knight", "Iron Man", "The Matrix", "Gravity",
        "Toy Story", "Finding Nemo"
    ],
    "genres": [
        "sci-fi space adventure",
        "sci-fi thriller action",
        "sci-fi space adventure",
        "action superhero adventure",
        "action crime thriller",
        "action superhero sci-fi",
        "sci-fi action thriller",
        "sci-fi space drama",
        "animation comedy family",
        "animation family adventure"
    ]
})

vectorizer = TfidfVectorizer(stop_words="english")
genre_matrix = vectorizer.fit_transform(movies["genres"])
similarity = cosine_similarity(genre_matrix)


def recommend(movie_name, number=5):
    matches = movies[movies["title"].str.lower() == movie_name.lower()]

    if matches.empty:
        print("Movie not found in the sample list.")
        return

    index = matches.index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda item: item[1], reverse=True)

    print(f"\nRecommendations for {movies.loc[index, 'title']}:")
    shown = 0

    for movie_index, score in scores:
        if movie_index == index:
            continue

        print(f"- {movies.loc[movie_index, 'title']}  (similarity: {score:.2f})")
        shown += 1

        if shown == number:
            break


print("Movie Recommendation System")
print("-" * 30)
print("Available movies:")
for title in movies["title"]:
    print("-", title)

choice = input("\nEnter a movie name: ").strip()
recommend(choice)
