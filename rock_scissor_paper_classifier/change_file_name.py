import os

# 해당 인덱스부터 차레대로 이름 지정
i = 0
# 변경하고자하는 파일이 있는 폴더 경로 적기
#  예시 : /home/ssac27/aiffel/rock_scissor_paper/paper"
file_path = "/home/ssac14/aiffel/rock_scissor_paper_test/scissor"

file_name = os.listdir(file_path)
for idx, name in enumerate(file_name):
    src = os.path.join(file_path, name)
    new_name = os.path.join(file_path, str(i+idx) +'.jpg')
    os.rename(src, new_name)