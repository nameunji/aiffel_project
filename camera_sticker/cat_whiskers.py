import os, time
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt


# 1. 얼굴 검출
# 1-1. 이미지 불러오기
img_path = os.getenv('HOME') + '/aiffel/camera_sticker/images/w5.jpg'
img_bgr = cv2.imread(img_path)  # 이미지 읽어서 저장
img_show = img_bgr.copy()  # 출력용 이미지 별도 보관
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print('img_size : ', img_rgb.shape)

# 1-2. detector를 선언하여 얼굴 영역 좌표 찾기
# 만약 얼굴이 여러개라면 len(dlib_rects) = n
detector_hog = dlib.get_frontal_face_detector()
start = time.time()  # 시작 시간 저장
dlib_rects = detector_hog(img_rgb, 1)  # 찾은 얼굴 영역 좌표 rectangles[[(27, 206) (348, 527)]]
print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
print(dlib_rects)
# -> gil.jpeg를 넣었을 때는 detector_hog의 upsample_num_times 인자를 5로 넣어줬음에도 시간이 10초정도 지나도 얼굴영역을 잡아내지 못했다.

# 1-3. 얼굴의 bounding box 그리기
for dlib_rect in dlib_rects:
    l = dlib_rect.left()
    t = dlib_rect.top()
    r = dlib_rect.right()
    b = dlib_rect.bottom()

    # check - face bounding box
    cv2.rectangle(img_show, (l, t), (r, b), (0, 255, 0), 2, lineType=cv2.LINE_AA)  # img_show에 라인 그리기

plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()

# 2. 얼굴 랜드마크 검출
# 2-1. 저장한 landmark 모델을 불러오기
model_path = os.getenv('HOME')+'/aiffel/camera_sticker/models/shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(model_path)

# 2-2. 개별 위치에 접근하여 (x,y)형태로 변환
list_landmarks = []
for dlib_rect in dlib_rects:
    points = landmark_predictor(img_rgb, dlib_rect)
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
    list_landmarks.append(list_points)

# check - face landmark
for landmark in list_landmarks:
    for idx, point in enumerate(list_points):
        cv2.circle(img_show, point, 2, (0, 255, 255), -1)  # yellow
plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()


# 3. 스티커 적용 위치 확인하기
# 3-1. 코의 좌표 확인 및 스티커의 위치 지정
for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
    # print(landmark[30])  # nose center index (187, 367)
    x = landmark[30][0]
    y = landmark[30][1]
    w = dlib_rect.width()
    h = dlib_rect.width()
    print('x, y, w, h : ', x, y, w, h)

# 3-2. 스티커 이미지 사이즈 수정
sticker_path = os.getenv('HOME')+'/aiffel/camera_sticker/images/cat-whiskers.png'
img_sticker = cv2.imread(sticker_path)
img_sticker = cv2.resize(img_sticker, (w, h))  # (322, 322, 3)
print(f'스티커 사이즈 : {img_sticker.shape}')

# 3-3. 스티커 이미지의 x,y좌표 조정. 이미지의 시작점은 top-left이기때문에.
refined_x = x - w // 2  # left
refined_y = y - h // 2  # top
print(f'(refined_x, refined_y) : ({refined_x}, {refined_y})')  # (109, 65)


# 4. 원본이미지에 스티커 적용
# 4-1. 스티커가 이미지를 벗어날 경우 잘라주기
if refined_x + w > img_show.shape[1]:  # 오른쪽으로 벗어날 경우
    img_sticker = img_sticker[:, :img_show.shape[1]-(refined_x+w)]
elif refined_x < 0:                    # 왼쪽으로 벗어날 경우
    img_sticker = img_sticker[:, -refined_x:]
    refined_x = 0

# 4-2. 스티커 이미지 영역 잡아주기
sticker_area = img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
print('sticker area : ', sticker_area.shape)

# 4-3. 스티커를 이미지에 적용하기
img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] \
    = np.where(img_sticker == 255, sticker_area, img_sticker).astype(np.uint8)
plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()