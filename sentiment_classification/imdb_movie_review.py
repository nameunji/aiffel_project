import os
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# 1. IMDB 데이터셋 다운로드
imdb = keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)  # num_words = 그 수만큼 word_to_index dict까지 생성된 형태로 데이터셋이 생성됨

# 2. 워드 사전 수정
word_to_index = imdb.get_word_index()

word_to_index = {k: (v+3) for k, v in word_to_index.items()}  # 형태 {'the': 4}
word_to_index["<PAD>"] = 0
word_to_index["<BOS>"] = 1
word_to_index["<UNK>"] = 2  # unknown
word_to_index["<UNUSED>"] = 3

index_to_word = {index: word for word, index in word_to_index.items()}
index_to_word[0] = "<PAD>"
index_to_word[1] = "<BOS>"
index_to_word[2] = "<UNK>"
index_to_word[3] = "<UNUSED>"
index_to_word = {index: word for word, index in word_to_index.items()}  # 형태 {4: 'the'}

# 3. maxlen(문장 최대 길이) 값 설정
total_data_text = list(x_train) + list(x_test)

# 3-1. 텍스트데이터 문장길이의 리스트를 생성한 후
num_tokens = [len(tokens) for tokens in total_data_text]  # 각각의 문장 길이를 담은 배열 생성
num_tokens = np.array(num_tokens)

# 3-2. 문장길이의 평균값, 최대값, 표준편차를 계산해 본다.
print('문장길이 평균 : ', np.mean(num_tokens))
print('문장길이 최대 : ', np.max(num_tokens))
print('문장길이 표준편차 : ', np.std(num_tokens))

# 3-4. 예를들어, 최대 길이를 (평균 + 2*표준편차)로 한다면,
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
maxlen = int(max_tokens)
print('pad_sequences maxlen : ', maxlen)
print('전체 문장의 {}%가 maxlen 설정값 이내에 포함됩니다. '.format(np.sum(num_tokens < max_tokens) / len(num_tokens)))

# 3-5. padding 방식(pre or post)
print(x_train.shape)
print(word_to_index["<PAD>"])
x_train = keras.preprocessing.sequence.pad_sequences(x_train, value=word_to_index["<PAD>"], padding='post', maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, value=word_to_index["<PAD>"], padding='post', maxlen=maxlen)
print(x_train.shape)

# 4. 어휘 사전 크기 & 워드벡터차원 수 설정
vocab_size = 10000    # 어휘 사전의 크기입니다(10,000개의 단어)
word_vector_dim = 16  # 워드 벡터의 차원수 (변경가능한 하이퍼파라미터)

# 5. model 설계 - 아래는 1-D CNN으로 설계했다.
# 1-D: 문장 전체를 한꺼번에 한 방향으로 길이 7짜리 필터로 스캐닝하면서 7단어 이내에서 발견되는 특징을 추출하여 그것으로 문장을 분류하는 방식으로 사용
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
model.add(keras.layers.Conv1D(16, 7, activation='relu'))
model.add(keras.layers.MaxPooling1D(5))
model.add(keras.layers.Conv1D(16, 7, activation='relu'))
model.add(keras.layers.GlobalMaxPooling1D())
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# 6. train set & valid set 분리
# 6-1. validation set 10000건 분리
x_val = x_train[:10000]
y_val = y_train[:10000]

# 6-2. validation set을 제외한 나머지 15000건(train)
partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]

# 7. 모델 학습
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
epochs = 20
history = model.fit(partial_x_train, partial_y_train, epochs=epochs, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# 8. 테스트
results = model.evaluate(x_test,  y_test, verbose=2)
print(results)

# 9. 결과 그래프화
history_dict = history.history
# print(history_dict.keys())  # epoch에 따른 그래프를 그려볼 수 있는 항목들 - ['loss', 'accuracy', 'val_loss', 'val_accuracy']

# 9-1. Training and validation loss - 몇 epoch까지의 트레이닝이 적절한지 최적점을 추정
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo"는 "파란색 점"입니다
plt.plot(epochs, loss, 'bo', label='Training loss')
# b는 "파란 실선"입니다
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 9-2. Training and validation accuracy
plt.clf()   # 그림을 초기화합니다

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()