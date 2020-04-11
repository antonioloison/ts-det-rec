"""Functions used to create the input matrix of the simple neural network
from the cropped image detected using polygons"""

import os
from PIL import Image
import numpy
import cv2

HELP_COMPONENTS = {"white": [[[0, 0, 100], [255, 100, 255]]], "black": [[[0, 0, 0], [255, 100, 100]]]}
SCALE_G = 0.5


def image2array(img):
    """
    Returns the image numpy array
    :param img: string of image path
    :return: numpy array of pixel intensities
    """
    """transforms image in ppm format to array"""
    img = Image.open(img)
    return numpy.array(img.getdata(), numpy.uint8).reshape(img.size[1], img.size[0], 3)


def show_image(img, title, scale=SCALE_G):
    """
    Shows the images
    :param img: image array
    :param title: string of the window title
    :param scale: float
    :return: None
    """
    newx, newy = int(img.shape[1] * scale), int(img.shape[0] * scale)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, (newx, newy))
    cv2.imshow(title, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def detect_image_component(image, components, show=False):
    """
    Takes an image and shows the mask of each components on components list
    :param image: image array
    :param components: component dictionnary
    :param show: bool, True if the image is shown
    :return: mask result
    """

    # convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # resulting list
    res = []
    # use of the threshold technique to select the mask for each component
    for component in components.keys():
        s = hsv.shape
        thresh = components[component]
        mask = numpy.zeros((s[0], s[1]))
        for threshold in thresh:
            lower_comp = numpy.array(threshold[0])
            higher_comp = numpy.array(threshold[1])
            mask += cv2.inRange(hsv, lower_comp, higher_comp)
        if show:
            show_image(mask, component)
        res.append(mask)
    return res


def reshape_masks(masks, category_number):
    """
    Concatenation of masks
    :param masks: mask list
    :param category_number: int
    :return: conctenated masks
    """
    """transforms the list of image_masks into a vector to vector"""
    final_mask = numpy.zeros((category_number, 1))
    for mask in masks:
        final_mask = numpy.concatenate((numpy.reshape(mask.T, [mask.shape[0] * mask.shape[1], 1]), final_mask), axis=0)
    return final_mask


def normalisation(image_path, category_number, components=HELP_COMPONENTS):
    """
    Normalizes image pixel intensites into values between  0 and 1
    :param image_path: string path of the image
    :param category_number: int
    :param components: dictionnary containing the interval of each mask
    :return: the normalized vector
    """
    matrix_img = image2array(image_path)
    masks = detect_image_component(matrix_img, components)
    vector = reshape_masks(masks, category_number)
    normalized_vector = vector / 255
    return normalized_vector


def load_input_with_shuffle(directory, first_image_path, category_number, categories):
    """
    Shuffles and normalizes the columns of the input matrix
    :param directory: string of the directory path
    :param first_image_path: string of the path of the first image in the directory
    :param category_number: int
    :param categories: list of categories
    :return: normalized and shuffled input matrix
    """
    final_matrix = normalisation(first_image_path, category_number)
    i = 0
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if i != 0:
                new_img = normalisation(directory + '\\' + filename, category_number)
                final_matrix = numpy.concatenate((final_matrix, new_img), axis=1)
            i += 1
    compteur = 0
    for root, dirs, files in os.walk(directory):
        for filename in files:
            for i in range(category_number):
                if categories[i] in filename:
                    final_matrix[(final_matrix.shape[0] - category_number + i), compteur] = 1
            compteur += 1
    final_matrix = final_matrix.T
    numpy.random.shuffle(final_matrix)
    return final_matrix.T

# Test
# print(load_input_with_shuffle(r"C:\Users\Antonio\Documents\Projet_Autonomous_Driving\Dataset_global\Dataset_entrainement_triangle\trianglesred_entrainement",
#       r"C:\Users\Antonio\Documents\Projet_Autonomous_Driving\Dataset_global\Dataset_entrainement_triangle\trianglesred_entrainement\attention49.jpg",
#       3)[1568:1571,0:20])
# #
# 1571, 354)
# (input.shape[0] - CATEGORY_NUMBER):input.shape[0], :
