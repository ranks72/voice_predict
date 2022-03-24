from flask import *
import pandas as pd

app = Flask(__name__)

# import dataset
movie_data = pd.read_csv('machine_learning_model/IMDb_Movie_Dataset.csv', index_col=0)
movie_rec_data = pd.read_csv('machine_learning_model/movie_recommendation_dataset.csv')

def get_recommendation(movie_name):
    # find movie index
    movie_index = movie_data.index[movie_data['Title'] == movie_name].values[0]

    # get list of movies recommendation
    movies = movie_rec_data.iloc[movie_index]
    movie_rec = []
    for movie in movies:
        temp = []
        title = movie_data['Title'][movie]
        year = movie_data['Year'][movie]
        genre = movie_data['Genre'][movie]
        rating = movie_data['Rating'][movie]

        temp.append(title)
        temp.append(year)
        temp.append(genre)
        temp.append(rating)
        movie_rec.append(temp)
    return movie_rec

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    input_movie = request.form["input"]

    # get input movie data
    input_movie_data = []
    input_movie_id = movie_data.index[movie_data['Title'] == input_movie].values[0]
    title = movie_data['Title'][input_movie_id]
    year = movie_data['Year'][input_movie_id]
    genre = movie_data['Genre'][input_movie_id]
    rating = movie_data['Rating'][input_movie_id]

    input_movie_data.append(title)
    input_movie_data.append(year)
    input_movie_data.append(genre)
    input_movie_data.append(rating)

    # get movie recommendation
    return render_template('home.html', movie_rec = get_recommendation(input_movie), input_movie_data = input_movie_data)

if __name__ == '__main__':
    app.run(debug=True)