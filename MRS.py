#importing libraries
import pandas as pd
import warnings
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

#ignore warnings
warnings.filterwarnings('ignore')

#importing csv files
movies = 'movies.csv'
ratings = 'ratings.csv'

#creating dataframe
df_movies = pd.read_csv(movies, usecols=['movieId','title'], dtype={'movieId':'int32','title':'str'})
df_ratings = pd.read_csv(ratings, usecols=['userId','movieId','rating'], dtype={'userId':'int32','movieId':'int32','rating':'float32'})

#creating sparce matrix
movies_users=df_ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
mat_movies_users=csr_matrix(movies_users.values)

#finding distance between matrices
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
model_knn.fit(mat_movies_users)

#recommendor function
def recommendor(movie_name, data, model):
    model.fit(data)
    idx = process.extractOne(movie_name,df_movies['title'])[2]
    print('Movie Selected: ', df_movies['title'][idx], 'Index: ',idx)
    print('Finding Recommendations')
    distances, indices = model.kneighbors(data[idx])
    for i in indices:
        print(df_movies['title'][i].where(i!=idx))

#main
print("Hello, please enter the name of the movie you want recommendations based on")
movie = input("Movie: ")
recommendor(movie, mat_movies_users, model_knn)
