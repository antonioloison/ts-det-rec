import os
from PIL import Image

# CONSTANTS
STAND_SIZE = (2592, 1944)
STAND_WIDTH = 1500
RATIO = STAND_SIZE[0] / STAND_SIZE[1]


# FORMAT CHANGE
def to_jpg(img_path):
    """
    Saves image to jpg format
    :param img_path: string path of the image
    :return: None
    """
    img = Image.open(img_path)
    file_name, _ = os.path.splitext(img_path)
    img.save(file_name + '.jpg')


def jp2jpg_directory(directory, new_directory):
    """
    Converts everything in jp2 format in directory to jpg format in new directory
    :param directory: string path of old directory
    :param new_directory: string path of new directory
    :return: None
    """
    i = 0
    print(i)
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if i % 100 == 0:
                print("Image treated: " + str(i))
            img = Image.open(directory + "\\" + filename)
            img.save(new_directory + "\image_" + str(i) + ".jpg")
            i += 1


# SIZE CHANGE
def resize(image_path):
    """
    Saves the image corresponding to the image path in the desired size by reshaping the image
    :param image_path: string path of the image to resize
    :return: None
    """
    img = Image.open(image_path)
    img = img.resize((STAND_WIDTH, int(STAND_WIDTH / RATIO)), Image.ANTIALIAS)
    fileName, fileExtension = os.path.splitext(image_path)
    img.save(fileName + fileExtension)


def crop(image_path, size=STAND_SIZE):
    """
    Saves image in the desired size by croping and resizing the image
    :param image_path: string path of the image to resize
    :param size: tuple size of the image
    :return: None
    """
    img = Image.open(image_path)
    width, height = img.size
    ratio = width / height
    stand_ratio = size[0] / size[1]
    if ratio > stand_ratio:
        # couper sur la longueur
        new_w = stand_ratio * height
        left = int((width - new_w) / 2)
        cropped_img = img.crop(((left, 0, width - left, height)))
    else:
        # couper sur hauteur
        new_h = width / stand_ratio
        bottom = int((height - new_h) / 2)
        cropped_img = img.crop(((0, height, width, bottom)))
    recropped_img = cropped_img.resize((size[0], size[1]), Image.ANTIALIAS)
    recropped_img.save(image_path)


def resize_directory(image_directory):
    """
    Resizes the images to have the bandwidth
    :param image_path: string path of the image directory
    :return: None
    """
    for root, dirs, files in os.walk(image_directory):
        for filename in files:
            resize(image_directory + '\\' + filename)


def directory_renamer(directory):
    """
    Renames images in directory
    :param image_path: string path of the image directory
    :return: None
    """
    i = 0
    for root, dirs, files in os.walk(directory):
        for filename in files:
            os.rename(directory + '\\' + filename, directory + '\\' + str(i) + ".ppm")
            i += 1
