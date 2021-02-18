import os
import cv2
import numpy as np
import urllib
import tarfile

from glob import glob
from matplotlib import pyplot as plt
import tensorflow as tf


img_path = os.getenv('HOME') + '/aiffel/human_segmentation/images/my_image.png'
img_orig = cv2.imread(img_path)


# segmentation으로 사람 분리하기
# semantic segmentation : '사람'이라는 추상적인 정보를 이미지에서 추출하는 방법, 그래서 사람이 누구인지 관계없이 같은 라벨로 표현됨.
# 모델 종류 : FCN, SegNet, U-Net, DeepLab 등
class DeepLabModel(object):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPrediction:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        self.graph = tf.Graph()
        graph_def = None
        tar_file = tarfile.open(tarball_path)

        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        with self.graph.as_default():
            tf.compat.v1.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.graph)

    # 이미지를 전처리하여 tf 입력으로 사용가능한 shape의 numpy array로 변환
    def preprocess(self, img_orig):
        height, width = img_orig.shape[:2]
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = cv2.resize(img_orig, target_size)
        resized_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        img_input = resized_rgb
        return img_input

    # segmentation 실행
    def run(self, image):
        img_input = self.preprocess(image)

        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [img_input]}
        )

        seg_map = batch_seg_map[0]
        return cv2.cvtColor((img_input, cv2.COLOR_RGB2BGR), seg_map)


_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'

model_dir = os.getenv('HOME') + '/aiffel/human_segmentation/models'
tf.io.gfile.makedirs(model_dir)

print('temp directory : ', model_dir)

# 구글이 제공하는 deeplabv3_mnv2_pascal_train_aug_2018_01_29 weight을 다운로드
# 이 모델은 PASCAL VOC 2012라는 대형 데이터셋으로 학습된 v3 버전
download_path = os.path.join(model_dir, 'deeplab_model.tar.gz')
if not os.path.exists(download_path):
    urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX+'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz', download_path)

MODEL = DeepLabModel(download_path)
print('model loaded successfully!')

img_resized, seg_map = MODEL.run(img_orig)
print(img_orig.shape, img_resized.shape, seg_map.max())  # (450, 800, 3) (288, 513, 3) 20

LABEL_NAMES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
]

img_show = img_resized.copy()
seg_map = np.where(seg_map == 15, 15, 0)  # 예측 중 사람만 추출 - 사람을 뜻하는 15 외 예측은 0으로 만듬, 그럼 예측된 seg이미지(map)은 최대값이 15가 된다.
img_mask = seg_map * (255/seg_map.max())  # 255 normalization - seg맵에 표현된 값을 원본 이미지에 그림형태로 출력하기위해 255로 정규화
img_mask = img_mask.astype(np.unit8)
color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)  # 색 적용
img_show = cv2.addWeighted(img_show, 0.6, color_mask, 0.35, 0.0)

plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()


# segmentation 결과 원래 크기로 복원
img_mask_up = cv2.resize(img_mask, img_orig.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
_, img_mask_up = cv2.threshold(img_mask_up, 128, 255, cv2.THRESH_BINARY)

ax = plt.subplot(1, 2, 1)
plt.imshow(img_mask_up, cmap=plt.cm.binary_r)
ax.set_title('Original Size Mask')

ax = plt.subplot(1, 2, 2)
plt.imshow(img_mask, cmap=plt.cm.binary_r)
ax.set_title('DeepLab Model Mask')

plt.show()


# 배경이미지 얻기
img_mask_color = cv2.cvtColor(img_mask_up, cv2.COLOR_GRAY2BGR)
img_bg_mask = cv2.bitwise_not(img_mask_color)    # 이미지 반전 - 배경 255, 사람 0
img_bg = cv2.bitwise_and(img_orig, img_bg_mask)  # 배경만 있는 영상 얻기
plt.imshow(img_bg)
plt.show()

# 배경 흐리게 하기
img_bg_blur = cv2.blur(img_bg, (13, 13))
plt.imshow(cv2.cvtColor(img_bg_blur, cv2.COLOR_BGR2RGB))
plt.show()







