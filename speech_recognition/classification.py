import os
import numpy as np
import librosa
import tensorflow as tf
import IPython.display as ipd
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# 1. 데이터 처리와 분류
# 1-1. 라벨 데이터 처리하기
# sklearn의 train_test_split함수를 이용하여 train, test 분리
data_path = os.getenv("HOME") + '/aiffel/data/speech_recognition/data/speech_wav_8000.npz'
speech_data = np.load(data_path)

label_value = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown', 'silence']
new_label_value = dict()
for i, l in enumerate(label_value):
    new_label_value[l] = i
label_value = new_label_value  # {'yes': 0, 'no': 1, 'up': 2, ...}

temp = []
for v in speech_data["label_vals"]:
    temp.append(label_value[v[0]])
label_data = np.array(temp)  # array([ 3,  3,  3, ..., 11, 11, 11])

# 1-2. train, test분리
train_wav, test_wav, train_label, test_label = train_test_split(speech_data["wav_vals"], label_data, test_size=0.1, shuffle=True)
# train_wav.shape = (45558, 8000)  train_wav[0].shape = (8000,)
del speech_data


# 2. 학습을 위한 하이퍼파라미터 설정
batch_size = 32
max_epochs = 10

# the save point - 후에 모델체크포인트 callback함수를 설정하거나 모델을 불러올 때 사용
checkpoint_dir = os.getenv('HOME') + '/aiffel/data/speech_recognition/models/wav'


# 3. 데이터셋 구성
# 3-1. 1차원의 Waveform 데이터를 2차원의 Spectrogram 데이터로 변환
def wav2spec(wav, fft_size=258):  # spectrogram shape을 맞추기위해서 size 변형
    D = np.abs(librosa.stft(wav, n_fft=fft_size))  # wav-파형의 amplitude 값, n_fft-win_length보다 길 경우 모두 zero padding해서 처리하기 위한 파라미터
    return D

def change_shape(waves):
    tmp_wav = []
    for wav in waves:
        tmp_wav.append(wav2spec(wav))
    return np.array(tmp_wav)

train_wav = change_shape(train_wav)  # (45558, 130, 126)
test_wav = change_shape(test_wav)    # (5062, 130, 126)

# 3-2. 차원 1 -> 2
# sr = 130
# sc = 126
# train_wav = train_wav.reshape(-1, sr, sc, 1)  # add channel (45558, 130, 126, 1)
# test_wav = test_wav.reshape(-1, sr, sc, 1)
# print(train_wav.shape, train_wav[0].shape)
# print(test_wav.shape, test_wav[0].shape)

# 3-3. 데이터 설정
# map : dataset이 데이터를 불러올때마다 동작시킬 데이터 전처리 함수를 매핑해줌
# one_hot : 단 하나의 값만 True로 (1) 이며 나머지는 모두 False(0)으로 된 Encoding / 주의 - 입력받은 행렬보다 마지막에 한차원이 증가
def one_hot_label(wav, label):
    label = tf.one_hot(label, depth=12)
    return wav, label

# for train
train_dataset = tf.data.Dataset.from_tensor_slices((train_wav, train_label))
train_dataset = train_dataset.map(one_hot_label)
train_dataset = train_dataset.repeat().batch(batch_size=batch_size)

# for test
test_dataset = tf.data.Dataset.from_tensor_slices((test_wav, test_label))
test_dataset = test_dataset.map(one_hot_label)
test_dataset = test_dataset.batch(batch_size=batch_size)


# 4. 모델 구성
input_tensor = layers.Input(shape=(130, 126))

x = layers.Conv1D(32, 9, padding='same', activation='relu')(input_tensor)
x = layers.Conv1D(32, 9, padding='same', activation='relu')(x)
x = layers.MaxPool1D()(x)

x = layers.Conv1D(64, 9, padding='same', activation='relu')(x)
x = layers.Conv1D(64, 9, padding='same', activation='relu')(x)
x = layers.MaxPool1D()(x)

x = layers.Conv1D(128, 9, padding='same', activation='relu')(x)
x = layers.Conv1D(128, 9, padding='same', activation='relu')(x)
x = layers.Conv1D(128, 9, padding='same', activation='relu')(x)
x = layers.MaxPool1D()(x)

x = layers.Conv1D(256, 9, padding='same', activation='relu')(x)
x = layers.Conv1D(256, 9, padding='same', activation='relu')(x)
x = layers.Conv1D(256, 9, padding='same', activation='relu')(x)
x = layers.MaxPool1D()(x)
x = layers.Dropout(0.3)(x)

x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

output_tensor = layers.Dense(12)(x)

model_wav = tf.keras.Model(input_tensor, output_tensor)

model_wav.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                                 save_weights_only=True,
                                                 monitor='val_loss',
                                                 mode='auto',
                                                 save_best_only=True,
                                                 verbose=1)

# 5. 학습
history_wav = model_wav.fit(train_dataset,
                            epochs=max_epochs,
                            steps_per_epoch=len(train_wav) // batch_size,
                            validation_data=test_dataset,
                            validation_steps=len(test_wav) // batch_size,
                            callbacks=[cp_callback]
                            )

# 6. 학습결과
acc = history_wav.history['accuracy']
val_acc = history_wav.history['val_accuracy']
loss = history_wav.history['loss']
val_loss = history_wav.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# 7. Evaluation 평가
model_wav.load_weights(checkpoint_dir)
results = model_wav.evaluate(test_dataset)

print("loss value: {:.3f}".format(results[0]))             # loss
print("accuracy value: {:.4f}%".format(results[1] * 100))  # accuracy

"""
학습결과
1. 1D - loss: 0.2656 - accuracy: 0.9194

"""