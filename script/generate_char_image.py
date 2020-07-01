from PIL import Image, ImageDraw, ImageFont
import numpy as np

font_path = "C:\\Windows\\Fonts\\meiryo.ttc"
tohu_array = np.asarray(Image.open("src/data/tohu.pbm")).astype(np.int32)


def generate_char_image(char):
    image_size = (60, 60)
    font_size = int(image_size[0] * 0.6)

    img = Image.new('1', image_size)
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(font_path, font_size)
    offset = tuple((si - sc) // 2 for si, sc in zip(image_size, font.getsize(char)))

    draw.text((offset[0], offset[1] // 2), char, fill=1, font=font)

    if np.all(tohu_array == np.asarray(img).astype(np.int32)):
        return

    img.save('src/data/char_image/{}.pbm'.format(ord(char)))


if __name__ == "__main__":
    with open("src/data/char_list.txt", "r", encoding="utf8") as f:
        lines = f.readlines()
    for char in lines[0]:
        generate_char_image(char)
