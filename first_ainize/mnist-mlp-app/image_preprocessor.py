from PIL import Image, ImageOps

"""
preprocess()
1. 이미지를 gray scale로 변경합니다.
2. 그 후 28 X 28의 크기로 사이즈 조정을 합니다.
3. 마지막으로 이미지의 색을 반전시킵니다.
"""
def preprocess(src, width, height):
   gray_image = src.convert('L')
   resized_image = gray_image.resize((width, height))
   dst = ImageOps.invert(resized_image)
   return dst