import os
import numpy as np

from PIL import Image
import face_recognition


dir_path = os.getenv('HOME')+'/aiffel/data/e7_face_embedding/image'
# pillow_path = os.getenv('HOME')+'/aiffel/data/e7_face_embedding/pillow/'
pillow_path = os.getenv('HOME')+'/aiffel/e7_face_embedding/static/'


# Step1. 얼굴 영역 자르기
# 얼굴 영역 구하는 함수
def get_gropped_face(image_file):
    image = face_recognition.load_image_file(image_file)
    face_locations = face_recognition.face_locations(image)

    a, b, c, d = face_locations[0]
    cropped_face = image[a:c, d:b, :]  # 이미지에서 얼굴영역만 잘라냄

    return cropped_face


# 얼굴 영역을 가지고 얼굴 임베딩 벡터를 구하는 함수
def get_face_embedding(face):
    return face_recognition.face_encodings(face)


def get_face_embedding_dict(path):
    file_list = os.listdir(path)
    embedding_dict = {}

    for file in file_list:
        file_name = file.split('.')[0]
        file_path = os.path.join(path, file)
        face = get_gropped_face(file_path)
        embedding = get_face_embedding(face)

        if len(embedding) > 0:  # 얼굴영역 face가 제대로 detect되지 않으면  len(embedding)==0인 경우가 발생하므로
            embedding_dict[file_name] = embedding[0]

            pillow_image = Image.fromarray(face)
            pillow_image_path = os.path.join(pillow_path, file)
            pillow_image.save(pillow_image_path)

    return embedding_dict


# 얼굴 벡터 전환
embedding_dict = get_face_embedding_dict(dir_path)


# 두 얼굴 임베딩 사이의 거리 구하기
def get_distance(name1, name2):
    return np.linalg.norm(embedding_dict[name1]-embedding_dict[name2], ord=2)


def get_sort_key_func(name1):
    def get_distance_from_name1(name2):
        return get_distance(name1, name2)

    return get_distance_from_name1


def get_nearest_face(name, top=5):
    sort_key_func = get_sort_key_func(name)
    sorted_faces = sorted(embedding_dict.items(), key=lambda x: sort_key_func(x[0]))
    result = []
    for i in range(top + 1):
        if i == 0: continue
        if sorted_faces[i]:
            print(f'순위 {i} : 이름({sorted_faces[i][0]}), 거리({sort_key_func(sorted_faces[i][0])})')
            result.append(sorted_faces[i][0])  # 닮은 사람 이름 반환하기 위해.
    return result


# get_nearest_face('남은지')

