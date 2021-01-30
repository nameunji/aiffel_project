import os
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from gensim.models import KeyedVectors
from tensorflow.keras.initializers import Constant


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# 1. IMDB 데이터셋 다운로드
imdb = keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# 2. 워드 사전 수정
word_to_index = imdb.get_word_index()
word_to_index = {k: (v+3) for k,v in word_to_index.items()}  # 형태 {'the': 4}
word_to_index["<PAD>"] = 0
word_to_index["<BOS>"] = 1
word_to_index["<UNK>"] = 2  # unknown
word_to_index["<UNUSED>"] = 3

index_to_word = {index:word for word, index in word_to_index.items()}
index_to_word[0] = "<PAD>"
index_to_word[1] = "<BOS>"
index_to_word[2] = "<UNK>"
index_to_word[3] = "<UNUSED>"
index_to_word = {index:word for word, index in word_to_index.items()}  # 형태 {4: 'the'}

# Google의 Word2Vec 모델 가져오기
# download : https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
word2vec_path = os.getenv('HOME')+'/aiffel/data/e4_sentiment_classification/GoogleNews-vectors-negative300.bin.gz'  # 300dim으로 이루어진 300만개 단어가 들어있음.
word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True, limit=1000000)  # 워드벡터 로딩(load_word2vec_format), limit으로 상위 100만개만 가져오기

vocab_size = 10000    # 어휘 사전의 크기입니다(10,000개의 단어)
word_vector_dim = 300  # 워드 벡터의 차원수 (변경가능한 하이퍼파라미터)

embedding_matrix = np.random.rand(vocab_size, word_vector_dim)
# np.random.rand(m,n) :  0-1사이의 난수를 matrix_array(m, n) 생성 m-행, n-열

# embedding_matrix에 Word2Vec 워드벡터를 단어 하나씩마다 차례차례 카피한다.
for i in range(4, vocab_size):
    if index_to_word[i] in word2vec:
        embedding_matrix[i] = word2vec[index_to_word[i]]  # ex) embedding_matrix[4] = word2vec['the']

total_data_text = list(x_train) + list(x_test)
# 텍스트데이터 문장길이의 리스트를 생성한 후
num_tokens = [len(tokens) for tokens in total_data_text]
num_tokens = np.array(num_tokens)
# 문장길이의 평균값, 최대값, 표준편차를 계산해 본다.
print('문장길이 평균 : ', np.mean(num_tokens))
print('문장길이 최대 : ', np.max(num_tokens))
print('문장길이 표준편차 : ', np.std(num_tokens))

# 예를들어, 최대 길이를 (평균 + 2*표준편차)로 한다면,
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
maxlen = int(max_tokens)
print('pad_sequences maxlen : ', maxlen)
print('전체 문장의 {}%가 maxlen 설정값 이내에 포함됩니다. '.format(np.sum(num_tokens < max_tokens) / len(num_tokens)))


# model 구성
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size,
                                 word_vector_dim,
                                 embeddings_initializer=Constant(embedding_matrix),  # 카피한 임베딩을 여기서 활용
                                 input_length=maxlen,
                                 trainable=True))   # trainable을 True로 주면 Fine-tuning
model.add(keras.layers.Conv1D(16, 7, activation='relu'))
model.add(keras.layers.MaxPooling1D(5))
model.add(keras.layers.Conv1D(16, 7, activation='relu'))
model.add(keras.layers.GlobalMaxPooling1D())
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# validation set 10000건 분리
x_val = x_train[:10000]
y_val = y_train[:10000]

# validation set을 제외한 나머지 15000건
partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]

# 모델 학습
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
epochs = 15  # 몇 epoch를 훈련하면 좋을지 결과를 보면서 바꾸어 봅시다.
history = model.fit(partial_x_train, partial_y_train, epochs=epochs, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# 테스트
results = model.evaluate(x_test,  y_test, verbose=2)
print(results)
