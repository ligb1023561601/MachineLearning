# _*_ coding: utf-8 _*_
# @Time     : 2017/9/28 19:27
# @Author    : Ligb
# @File     : CharImage.py
"""pillow 库将图像转换为字符画"""

from PIL import Image

# image_path = "test.PNG"
image_path = r'E:\PY_Coding\KNN\datas\0.png'
txt_path = "char.txt"

# ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,/'^`'.")
ascii_char = list("10")


def get_char(r, g, b, alpha=256):
    """将对应的灰度值转换为相应的字符"""
    length = len(ascii_char)
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    unit = 257 / length
    return ascii_char[int(gray/unit)]

if __name__ == '__main__':
    png_image = Image.open(image_path)
    png_image = png_image.resize((png_image.width, png_image.height), Image.NEAREST)
    txt_char = ""
    for y in range(png_image.height):
        for x in range(png_image.width):
            # *代表将返回的列表或元组拆分
            txt_char += get_char(*png_image.getpixel((x,y)))
        txt_char += "\n"
    print(txt_char)
    with open(txt_path, 'w') as char_image:
        char_image.write(txt_char)



