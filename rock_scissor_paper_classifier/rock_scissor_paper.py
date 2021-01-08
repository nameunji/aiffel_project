import os, glob
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow import keras
from sklearn.model_selection import train_test_split


# GPU out of memory 문제로 코드 추가
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


# 모든 파일 사이즈를 동일하게 28*28사이즈로 맞춰준다.
def change_file_size(path):
    image_dir_path = os.getenv("HOME") + "/aiffel_project/rock_scissor_paper_classifier/" + path
    images = glob.glob(image_dir_path + "/*.jpg")
    target_size = (28, 28)

    for img in images:
        old_img = Image.open(img)
        new_img = old_img.resize(target_size, Image.ANTIALIAS)
        new_img.save(img, "JPEG")

    print(f"{path} 이미지 resize 완료!")


# train, test data의 사이즈를 변경해주는 함수
# 테스트할 때 1번만 진행되면 되기 때문에 위해 함수로 묶어주었다.
def resize_train_test():
    for x in ['scissor', 'rock', 'paper']:
        train_path = 'train/' + x
        test_path = 'test/' + x
        change_file_size(train_path)
        change_file_size(test_path)


# 폴더별 가위 0, 바위 1, 보 2 라벨링
def load_data(img_path, len_data):
    # 가위 : 0, 바위 : 1, 보 : 2
    number_of_data = len_data  # 가위바위보 이미지 개수 총합에 주의하세요.
    img_size = 28
    color = 3

    # 이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs = np.zeros(number_of_data * img_size * img_size * color, dtype=np.int32).reshape(number_of_data, img_size, img_size, color)
    labels = np.zeros(number_of_data, dtype=np.int32)

    idx = 0
    for file in glob.iglob(img_path + '/scissor/*.jpg'):
        img = np.array(Image.open(file), dtype=np.int32)
        imgs[idx, :, :, :] = img  # 데이터 영역에 이미지 행렬을 복사
        labels[idx] = 0  # 가위 : 0
        idx = idx + 1

    for file in glob.iglob(img_path + '/rock/*.jpg'):
        img = np.array(Image.open(file), dtype=np.int32)
        imgs[idx, :, :, :] = img  # 데이터 영역에 이미지 행렬을 복사
        labels[idx] = 1  # 바위 : 1
        idx = idx + 1

    for file in glob.iglob(img_path + '/paper/*.jpg'):
        img = np.array(Image.open(file), dtype=np.int32)
        imgs[idx, :, :, :] = img  # 데이터 영역에 이미지 행렬을 복사
        labels[idx] = 2  # 보 : 2
        idx = idx + 1

    print("데이터의 이미지 개수는", idx, "입니다.")
    return imgs, labels


# train data를 라벨링한 후 정규화 시켜준다.
image_dir_path = os.getenv("HOME") + "/aiffel_project/rock_scissor_paper_classifier/train"
(x_train, y_train) = load_data(image_dir_path, 2370)
x_train_norm = x_train/255.0   # 입력은 0~1 사이의 값으로 정규화

# train data를 train, valid로 나눈다.(비율 70:30)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_norm, y_train, test_size=0.3, random_state=25)

# test data를 라벨링한 후 정규화 시켜준다.
image_dir_path = os.getenv("HOME") + "/aiffel_project/rock_scissor_paper_classifier/test"
(x_test, y_test) = load_data(image_dir_path, 990)
x_test_norm = x_test/255.0   # 입력은 0~1 사이의 값으로 정규화


# 모델 설계
n_channel_1 = 384
n_channel_2 = 128  # 128: loss: 0.9009 - accuracy: 0.6657
n_dense = 30

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(n_channel_1, (3,3), activation='relu', input_shape=(28,28,3)))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Conv2D(n_channel_2, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(n_dense, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# 모델 학습
n_train_epoch = 10
model.fit(x_train, y_train, epochs=n_train_epoch, validation_data=(x_valid, y_valid), batch_size=100)

# 모델 평가하기
test_loss, test_accuracy = model.evaluate(x_test_norm, y_test, verbose=2)  # verbose/ 0:silent 1:progress bar 2:one line per epoch
print("test_loss: {} ".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))