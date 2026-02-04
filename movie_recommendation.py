import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")

vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies["genre"])

similarity = cosine_similarity(genre_matrix)

def recommend(movie_title):
    if movie_title not in movies["title"].values:
        print("Movie not found!")
        return

    index = movies[movies["title"] == movie_title].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("Recommended movies:")
    for i in scores[1:4]:
        print(movies.iloc[i[0]]["title"])

movie_name = input("Enter movie name: ")
recommend(movie_name)
