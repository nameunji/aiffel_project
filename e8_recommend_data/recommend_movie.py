import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

# 1) 데이터 준비와 전처리
rating_file_path = os.getenv('HOME') + '/aiffel/data/recommend_data/data/ml-1m/ratings.dat'
ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv(rating_file_path, sep='::', names=ratings_cols, engine='python')
orginal_data_size = len(ratings)

# 3점 이상만 남긴다.
ratings = ratings[ratings['rating'] >= 3]
filtered_data_size = len(ratings)
print(f'orginal_data_size: {orginal_data_size}, filtered_data_size: {filtered_data_size}')
print(f'Ratio of Remaining Data is {filtered_data_size / orginal_data_size:.2%}')

# rating 컬럼의 이름을 count로 바꾼다. 별점 -> 시청횟수
ratings.rename(columns={'rating': 'count'}, inplace=True)

# 영화 제목을 보기 위해 메타 데이터를 읽어온다.
movie_file_path = os.getenv('HOME') + '/aiffel/data/recommend_data/data/ml-1m/movies.dat'
cols = ['movie_id', 'title', 'genre']
movies = pd.read_csv(movie_file_path, sep='::', names=cols, engine='python')

# 2) 분석
print(ratings['movie_id'].nunique())  # ratings에 있는 유니크한 영화 개수
print(ratings['user_id'].nunique())  # rating에 있는 유니크한 사용자 수
rating_count = ratings.groupby('movie_id')['count'].count()
print(rating_count.sort_values(ascending=False).head(10))  # 가장 인기 있는 영화 10개(인기순)


# 3) 내가 선호하는 영화를 5가지 골라서 rating에 추가
# ratings : user_id, movie_id, count, timestamp
# movies : movie_id, title, genre

# 내가 선호하는 영화가 movies에 있는지 체크
def check_movie(my_movies, movies):
    return [True if (movies['title'] == x).any() else False for x in my_movies]


my_movies_title = ['Dead Poets Society (1989)', 'Good Will Hunting (1997)', 'Toy Story (1995)', 'Notting Hill (1999)',
                   'Forrest Gump (1994)']


# print(check_movie(my_movies_title, movies))

# 영화리스트를 index로 변환
def title2index(my_movies, movies):
    return [movies[movies['title'] == movie]['movie_id'].values[0] for movie in my_movies]


my_movies_index = title2index(my_movies_title, movies)
# 사용자 인덱스 생성
my_id = ratings['user_id'].max() + 1
# pandas dataframe으로 바꿔줌
my_movie_list = pd.DataFrame(
    {'user_id': [my_id] * 5, 'movie_id': my_movies_index, 'count': [5.0] * 5, 'timestamp': [956715648] * 5})

# ratings에 추가
if not ratings.isin({'user_id': [my_id]})['user_id'].any():
    ratings = ratings.append(my_movie_list)

# 4) CSR matrix 생성
# csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
n_user = ratings['user_id'].nunique()
n_movie = ratings['movie_id'].nunique()
# print(n_user, n_movie)     # 6040, 3628

# csr_data = csr_matrix((ratings.count, (ratings.user_id, ratings.movie_id)), shape=(n_user, n_movie))
csr_data = csr_matrix((ratings['count'], (ratings.user_id, ratings.movie_id)))

# 5) als_model = AlternatingLeastSquares 모델을 직접 구성하여 훈련
als_model = AlternatingLeastSquares(factors=300, regularization=0.01, use_gpu=False, iterations=20, dtype=np.float32)
csr_data_transpose = csr_data.T  # als 모델은 input으로 (item X user 꼴의 matrix를 받기 때문에 Transpose해준다.
als_model.fit(csr_data_transpose)  # 모델 훈련

# 6) 내가 선호하는 5가지 영화 중 하나와 그 외의 영화 하나를 골라 훈련된 모델이 예측한 나의 선호도를 파악
# movie_id로 영화 title가져오기
def get_movie_name(idx):
    if idx in movies.movie_id:
        return movies[movies['movie_id'] == idx]['title'].values[0]
    else:
        print('해당 인덱스의 영화가 존재하지 않습니다.')

# 내 벡터와 영화 포레스트검프의 벡터가져오기
forrest_gump_id = my_movies_index[4]
my_vector, forrest_gump_vector = als_model.user_factors[my_id], als_model.item_factors[my_movies_index[4]]
# my_vector와 forrest_gump_vector를 내적하는 코드
a = np.dot(my_vector, forrest_gump_vector)

# my_vector와 father_of_the_bride_vector를 내적하는 코드
father_of_the_bride_id = 5
father_of_the_bride_vector = als_model.item_factors[father_of_the_bride_id]
b = np.dot(my_vector, father_of_the_bride_vector)

print(f'내가 선호하는 영화 {get_movie_name(forrest_gump_id)}와의 선호도 : {a}')
print(f'그 외의 영화 {get_movie_name(father_of_the_bride_id)}와의 선호도 : {b}')

# 7) 내가 좋아하는 영화와 비슷한 영화를 추천
def get_similar_movie(movie_title, movies, n=10):
    exist_movie = check_movie([movie_title], movies)[0]

    if exist_movie:
        movie_id = title2index([movie_title], movies)[0]
        similar_movie = als_model.similar_items(movie_id, N=n)
        similar_movie = [get_movie_name(i[0]) for i in similar_movie]
        return similar_movie

    print('해당 영화가 데이터에 없습니다.')
    return None


print(get_similar_movie(my_movies_title[0], movies))

# 8) 내가 가장 좋아할 만한 영화들을 추천
movie_recommended = als_model.recommend(my_id, csr_data, N=20, filter_already_liked_items=True)
print([get_movie_name(i[0]) for i in movie_recommended])

# 추천 기여도 확인
recommend_movie_id = movie_recommended[0][0]
explain = als_model.explain(my_id, csr_data, itemid=recommend_movie_id)
print([(get_movie_name(i[0]), i[1]) for i in explain[1]])
