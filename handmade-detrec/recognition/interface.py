"""
Interface functions to visualise the results
"""

import sys

sys.path.insert(0, r'C:\Users\Antonio\Documents\Projet_Autonomous_Driving\aadc\Detection_panneaux')

import Detection_module as dm
from neural_network import *
from input_creator import *
import cv2
import numpy as np

# CONSTANTS
BASEWIDTH = 28
HELP_COMPONENTS = {"white": [[[0, 0, 100], [255, 100, 255]]], "black": [[[0, 0, 0], [255, 100, 100]]]}
SCALE_G = 0.5
POLYGON_CATEGORIES = {"triangles": {"red": ["attention", "priorite", "autre"], "blue": []},
                      "rectangles": {"red": [], "blue": ["parking", "passage_p", "autre"]},
                      "circles": {"red": ["limvit20", "limvit30", "stop", "autre"],
                                  "blue": ["fleche_d", "fleche_g", "autre"]}}
TEXT_H, TEXT_W = 17, 70
VALID_CATEGORIES = ["trianglesred", "rectanglesblue", "circlesred", "circlesblue"]


def show_image(img, title, scale=SCALE_G):
    """
    Shows image with the right window size
    :param img: image numpy array
    :param title: string of the window title
    :param scale: image visualisation scale
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
    Returns the list of masks of the image
    :param image: image numpy array
    :param components: dictionnary containing the interval of each mask
    :param show: bool True if the image is shown
    :return: list of numpy arrays
    """

    # convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # resulting list
    res = []
    # use of the threshold technique to select the mask for each component
    for component in components.keys():
        s = hsv.shape
        thresh = components[component]
        mask = np.zeros((s[0], s[1]))
        for threshold in thresh:
            lower_comp = np.array(threshold[0])
            higher_comp = np.array(threshold[1])
            mask += cv2.inRange(hsv, lower_comp, higher_comp)
        if show:
            show_image(mask, component)
        res.append(mask)
    return res


def resize(image, basewidth=BASEWIDTH):
    """
    Returns the resized image
    :param image: image numpy array
    :param basewidth: image width
    :return: resized array
    """
    return cv2.resize(image, dsize=(basewidth, basewidth), interpolation=cv2.INTER_CUBIC)


def reshape_masks(masks, category_number):
    """
    Transforms the list of image_masks into a vector to vector
    :param masks: list of masks
    :param category_number: int
    :return: vecctor with concatenated masks
    REFACTOR WITH NORMALISATION IN INPUT CREATOR
    """
    final_mask = np.reshape(masks[0].T, [masks[0].shape[0] * masks[0].shape[1], 1])
    for i in range(1, len(masks)):
        mask = masks[i]
        final_mask = np.concatenate((np.reshape(mask.T, [mask.shape[0] * mask.shape[1], 1]), final_mask), axis=0)

    return final_mask


def normalisation(matrix_img, category_number, components=HELP_COMPONENTS):
    """
    REFACTOR WITH NORMALISATION IN INPUT CREATOR
    """
    masks = detect_image_component(matrix_img, components)
    vector = reshape_masks(masks, category_number)
    normalized_vector = vector / 255
    return normalized_vector


def draw_object(image, object_name, location, category):
    """
    Draws square around object and displays its name in image
    :param image: image array
    :param object_name: string name of the highlighted object
    :param location: tuple (x,y, height, width)
    :param category: object category
    :return: None
    """
    (x, y, h, w) = location
    if category == "trianglesred":
        color = (255, 0, 0)
    elif category == "rectanglesblue":
        color = (0, 255, 0)
    elif category == "circlesred":
        color = (0, 0, 255)
    elif category == "circlesblue":
        color = (0, 255, 255)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.fillConvexPoly(image, np.array(
        [[[x - 2, y - TEXT_H]], [[x - 2 + TEXT_W, y - TEXT_H]], [[x - 2 + TEXT_W, y]], [[x - 2, y]]]), color=color)
    cv2.putText(image, object_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)


def detected_signs(image, nns, show=False):
    """
    Locates every sign in image
    :param image: image numpy array
    :param nns: neural network dictionnary with keys as category names and values as the respective neural networks
    :param show: bool, show image if True
    :return: dictionnary with deteted signs in image
    """

    classed_polygons = dm.easy_give_signs(image)
    output_categories = {"triangles": {"red": [], "blue": []},
                         "rectangles": {"red": [], "blue": []},
                         "circles": {"red": [], "blue": []}}
    for polygon_cat in classed_polygons.keys():
        for color in classed_polygons[polygon_cat].keys():
            cla_col = polygon_cat + color
            if cla_col in VALID_CATEGORIES:
                polygons = classed_polygons[polygon_cat][color]
                for polygon in polygons:
                    location, img_polygon = polygon
                    # find the category
                    resized_img = resize(img_polygon, basewidth=BASEWIDTH)
                    neural_net = nns[cla_col]
                    category_nbre = neural_net.layers[-1]
                    input_vect = normalisation(resized_img, category_nbre)
                    output_vect = neural_net.calculate(input_vect)
                    argmaxi = np.argmax(output_vect)
                    # for future functions
                    category = POLYGON_CATEGORIES[polygon_cat][color][argmaxi]
                    output_categories[polygon_cat][color].append(category)

                    # draw the rectangle around the object
                    if category != "autre":
                        draw_object(image, category, location, cla_col)

    if show:
        show_image(image, "recognized_signs")

    return output_categories
