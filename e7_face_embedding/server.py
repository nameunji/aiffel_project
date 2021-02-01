"""
업로드한 파일을 가지고 닮은 연예인 찾기
"""

import os
import fnmatch
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from e7_face_embedding.face_embedding import embedding_dict, get_face_embedding_dict, get_nearest_face
from collections import defaultdict

UPLOAD_DIR = os.getenv('HOME')+'/aiffel/data/e7_face_embedding/upload'
pillow_path = os.getenv('HOME')+'/aiffel/e7_face_embedding/static/'

app = Flask(__name__)
app.config['UPLOAD_DIR'] = UPLOAD_DIR

# 업로드된 이미지 작업
def check(fname):
    file_name = fname.split('.')[0]  # 파일명
    embedding = get_face_embedding_dict(UPLOAD_DIR)  # 업로드 이미지의 임베딩벡터 만들기
    img_link = defaultdict(dict)  # 업로드이미지와 연예인이미지 링크 담기
    if embedding:
        embedding_dict.update(embedding)  # 기존 임베딩 벡터에 업로드이미지 벡터 추가
        nearest_star = get_nearest_face(file_name, top=1)  # 가장 닮은 연예인 가져오기
        nearest_star = nearest_star[0]
        # 링크 가져오기
        for file in os.listdir(pillow_path):
            if fnmatch.fnmatch(file, f'{nearest_star}.*'):
                img_link['star']['link'] = file
                img_link['star']['name'] = nearest_star
                break
    return img_link

# 파일 업로드를 위한 html 렌더링
@app.route('/')
def upload_main():
    return render_template("index.html")

# 서버에 파일 업로드
@app.route('/file-upload', methods=['POST'])
def upload_files():
    if request.method == 'POST':
        f = request.files['file']
        fname = secure_filename(f.filename)  # 사용자가 서버의 파일시스템이 있는 파일을 수정하는 것을 방지
        f.save(os.path.join(app.config['UPLOAD_DIR'], fname))
        img_link = check(fname)
        if img_link:
            return render_template("success.html", img_link=img_link)
        else:
            return render_template("fail.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
