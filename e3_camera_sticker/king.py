import os
import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np

# 편의를 위해 이미지 크기를 변경
my_image_path = os.getenv('HOME') + '/aiffel/data/e3_camera_sticker/images/image.jpg'
img_bgr = cv2.imread(my_image_path)  # OpenCV로 이미지를 읽어서
# img_bgr = cv2.resize(img_bgr, (234, 416))  # 640x360의 크기로 Resize
img_show = img_bgr.copy()  # 출력용 이미지 별도 보관 type : <class 'numpy.ndarray'>
plt.imshow(img_bgr)
plt.show()  # opencv은 BGR을 사용하기 때문에, 붉은색은 푸른색으로, 푸른색은 붉은색으로 바뀌어 출력된 것.

# plt.imshow 이전에 RGB 이미지로 바꾸는 것을 잊지마세요.
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()

detector_hog = dlib.get_frontal_face_detector()  # detector 선언  <_dlib_pybind11.fhog_object_detector object at 0x7fdf6d71b1f0>
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # dlib은 rgb 이미지를 입력으로 받기 때문에 cvtColor() 를 이용해서 opencv 의 bgr 이미지를 rgb로 변환
dlib_rects = detector_hog(img_rgb, 1)  # (image, num of img pyramid) -> 찾은 얼굴영역 좌표


# 찾은 얼굴 화면에 출력
print(dlib_rects)   # 찾은 얼굴영역 좌표 rectangles[[(201, 222) (386, 407)]] [(left, top), (right, bottom)]

for dlib_rect in dlib_rects:
    # dlib.rectangle객체는 left(), top(), right(), bottom(), height(), width() 등을 포함하고 있다.
    l = dlib_rect.left()
    t = dlib_rect.top()
    r = dlib_rect.right()
    b = dlib_rect.bottom()

    # left, top, right, bottom 변수에 담은 후 img_show에 라인 그리기
    cv2.rectangle(img_show, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA)
    # rectangle : 두 개의 반대쪽 모서리가있는 사각형 외곽선 또는 채워진 사각형을 그립니다.
    # rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None)
    # pt1 = 직사각형의 꼭지점
    # pt2 = pt1 반대쪽 직사각형의 꼭지점
    # color = 직사각형 색상 또는 밝기 (회색조 이미지)
    # thickness = 직사각형을 구성하는 선의 두께
    # lineType = 라인 유형(위에서 cv2.LINE_AA = 16)

img_show_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()


# 저장한 landmark 모델을 불러오기
model_path = os.getenv('HOME')+'/aiffel/data/e3_camera_sticker/models/shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(model_path)

list_landmarks = []
for dlib_rect in dlib_rects:
    points = landmark_predictor(img_rgb, dlib_rect)   # landmark_predictor 는 (RGB 이미지, dlib.rectangle)을 받아 dlib.full_object_detection를 반환
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))  # 개별 위치에 접근하여 (x,y)형태로 변환
    list_landmarks.append(list_points)

print(len(list_landmarks[0]))  # 68 이미지에서 찾아진 얼굴 개수마다 반복하면 list_landmark에 68개의 랜드마크가 얼굴 개수만큼 저장됩니다.

# 랜드마크를 영상에 출력
for landmark in list_landmarks:
    for idx, point in enumerate(list_points):
        cv2.circle(img_show, point, 2, (0, 255, 255), -1)  # yellow

img_show_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()

# 코의 좌표 확인 및 스티커의 위치 지정
for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
    print(landmark[30])  # nose center index : 30   (x, y) = (286, 311)
    x = landmark[30][0]  # 스티커 x좌표
    y = landmark[30][1] - dlib_rect.width()//2
    w = dlib_rect.width()
    h = dlib_rect.width()
    print(f'(x,y) : ({x},{y})')  # (x,y) : (286,218)
    print(f'(w,h) : ({w},{h})')  # (w,h) : (186,186)

# 스티커 이미지를 읽어서 사이즈 적용
sticker_path = os.getenv('HOME')+'/aiffel/data/e3_camera_sticker/images/king.png'
img_sticker = cv2.imread(sticker_path)
img_sticker = cv2.resize(img_sticker, (w, h))
print(img_sticker.shape)  # (186, 186, 3)

# 원본 이미지에 스티커 이미지를 추가하기 위해서 x, y 좌표를 조정합니다. 이미지 시작점은 top-left 좌표이기 때문입니다.
refined_x = x - w // 2  # left
refined_y = y - h       # top
print(f'(x,y) : ({refined_x},{refined_y})')  # (x,y) : (193,32)
# y축 좌표가 음수가 될 수 있다. 하지만 이미지 범위 밖의 -인덱스는 접근할 수 없기 때문에, 음수에 대한 예외처리를 해주어야 한다.
print(img_sticker.shape)  # (186, 186, 3)
# img_sticker = img_sticker[-refined_y:]  # 원본 이미지의 범위를 벗어난 스티커 부분을 제거(-y 크기만큼 스티커를 crop) # 범위를 벗어나지 않았기 때문에 굳이 잘라주지 않음.
print(img_sticker.shape)  # (32, 186, 3)  # (186, 186, 3)

# top 의 y 좌표는 원본 이미지의 경계 값으로 수정
refined_y = 0
print(f'(x,y) : ({refined_x},{refined_y})')  # (193,0)

# 원본 이미지에 스티커 적용
# sticker_area = 원본이미지에서 스티커를 적용할 위치를 crop한 이미지
# 예제에서는 (566,0) 부터 (566+268, 0+157) 범위의 이미지를 의미합니다.
sticker_area = img_show[refined_y:img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]

# 스티커 이미지에서 사용할 부분은 0 이 아닌 색이 있는 부분을 사용합니다. 따라서 np.where를 통해 img_sticker 가 0 인 부분은 sticker_area를 사용하고 0이 아닌 부분을 img_sticker를 사용하시면 됩니다
img_show[refined_y:img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] \
    = np.where(img_sticker == 0, sticker_area, img_sticker).astype(np.uint8)
# img_show[0:186, 193:379(193+186)] = np.where(?,?,?).astype(np.uint8)

# 출력
plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()
