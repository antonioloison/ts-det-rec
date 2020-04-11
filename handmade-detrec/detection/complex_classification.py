'''This is a more complex way of classification that would be used in the detection module but
doesn't work very well for now and must be perfected'''

import preprocessing_module as pm
from detection_module import *
import numpy as np
import cv2

HELP_COMPONENTS = {"white": [[[0, 0, 100], [255, 100, 255]]], "black": [[[0, 0, 0], [255, 80, 60]]]}

def verif_triangle(triangle, image, show=False):
    """
    Classify the triangle depending on the colour of the contours inside
    the triangle
    :param triangle: triangle coordinates list
    :param image: image array
    :param show: bool to show
    :return: triangle class
    """
    white_edges = pm.image_contour(pm.detect_image_component(image, HELP_COMPONENTS, show=show)[0])
    white_contours = cv2.findContours(white_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for w_contour in white_contours:
        if is_contained(w_contour, triangle):
            black_edges = pm.image_contour(pm.detect_image_component(image, HELP_COMPONENTS, show=show)[1])
            black_contours = cv2.findContours(black_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            for b_contour in black_contours:
                if is_contained(b_contour, triangle):
                    return "rwb"
            return "rw"
    return "nothing"

def verif_rectangle(rectangle, image, show=False):
    """
    Classify the rectangle depending on the colour of the contours inside
    the rectangle
    :param triangle: rectangle coordinates list
    :param image: image array
    :param show: bool to show
    :return: rectangle class
    """
    white_edges = pm.image_contour(pm.detect_image_component(image, HELP_COMPONENTS, show=show)[0])
    white_contours = cv2.findContours(white_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for w_contour in white_contours:
        if is_contained(w_contour, rectangle):
            return "bw"
    return "nothing"


def verif_circle(circle, color, image, show=False):
    """
    Classify the circle depending on the colour of the contours inside
    the circle
    :param triangle: circle coordinates list
    :param image: image array
    :param show: bool to show
    :return: circle class
    """
    comps = pm.detect_image_component(image, HELP_COMPONENTS, show=show)
    white_edges = pm.image_contour(comps[0], show=show)
    white_contours = cv2.findContours(white_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    if color:
        for w_contour in white_contours:
            if is_contained(w_contour, circle):
                black_edges = pm.image_contour(comps[1])
                black_contours = cv2.findContours(black_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                for b_contour in black_contours:
                    if is_contained(b_contour, circle):
                        return "rwb"
                return "rw"
    else:
        for w_contour in white_contours:
            if is_contained(w_contour, circle):
                return "bw"
    return "nothing"


def classify(image, triangles, rectangles, circles, circle_color):
    """
    Takes the three kinds of signs and the circle base color
    which is 0 for blue and 1 for red and classifies them
    :param image: image array
    :param triangles: coordinates list
    :param rectangles: coordinates list
    :param circles: coordinates list
    :param circle_color -> int: 0 for blue and 1 for red
    :return: tuple of classified polygons
    """
    classified_triangles = {"rwb": [], "rw": [], "nothing": []}
    classified_rectangles = {"bw": [], "nothing": []}
    if circle_color:
        classified_circles = {"bw": [], "nothing": []}
    else:
        classified_circles = {"rwb": [], "rw": [], "nothing": []}
    for triangle in triangles:
        (x, y), h, w = capt_rectangle(triangle)
        crop_img = image[y:y + h, x:x + w]
        ratio = h / w
        translated_triangle = triangle - np.tile([[x, y]], (triangle.shape[0], 1, 1))
        clas = verif_triangle(translated_triangle, crop_img)
        if 2 > ratio > 0.5:
            classified_triangles[clas].append(triangle)
            if clas != "nothing":
                cv2.polylines(image, [np.array([[[x, y]], [[x, y + h]], [[x + w, y + h]], [[x + w, y]]])], True,
                              (255, 0, 0), thickness=2)
    for rectangle in rectangles:
        (x, y), h, w = capt_rectangle(rectangle)
        crop_img = image[y:y + h, x:x + w]
        ratio = h / w
        translated_rectangle = rectangle - np.tile([[x, y]], (rectangle.shape[0], 1, 1))
        clas = verif_rectangle(translated_rectangle, crop_img)
        if 2 > ratio > 0.5:
            classified_rectangles[clas].append(rectangle)
            if clas != "nothing":
                cv2.polylines(image, [np.array([[[x, y]], [[x, y + h]], [[x + w, y + h]], [[x + w, y]]])], True,
                              (0, 255, 0), thickness=2)
    for circle in circles:
        (x, y), h, w = capt_rectangle(circle)
        crop_img = image[y:y + h, x:x + w]
        ratio = h / w
        translated_circle = circle - np.tile([[x, y]], (circle.shape[0], 1, 1))
        clas = verif_circle(translated_circle, circle_color, crop_img)
        if 2 > ratio > 0.5:
            classified_circles[clas].append(circle)
            if clas != "nothing":
                cv2.polylines(image, [np.array([[[x, y]], [[x, y + h]], [[x + w, y + h]], [[x + w, y]]])], True,
                              (0, 0, 255), thickness=2)
    return classified_triangles, classified_rectangles, classified_circles


def montre_panneau_verif(image_path, title="", verify=True):
    """
    shows the image with the highlighted traffic signs and
    by verifying that the rectangle containing the sign
    is not too flat and classes them if verify
    :param image_path: string path of the image
    :param title: string of the window title
    :param verify: bool indicating if it checks polygon flatness
    :return: None
    """
    img = cv2.imread(image_path)
    image_masks = pm.each_image(pm.detect_image_component(img, detection_components), pm.image_contour)
    masks_polygons = pm.each_image(image_masks, find_shape, i=1)
    i = 0
    for polygons in masks_polygons:
        triangles, rectangles, circles = interesting_polygon(polygons)
        if verify:
            classified_triangles, classified_rectangles, classified_circles = classify(img, triangles, rectangles, circles, i)
        else:
            for triangle in triangles:
                (x, y), h, w = capt_rectangle(triangle)
                ratio = h / w
                if 2 > ratio > 0.5:
                    cv2.polylines(img, [np.array([[[x, y]], [[x, y + h]], [[x + w, y + h]], [[x + w, y]]])], True,
                                  (255, 0, 0), thickness=2)
            for rectangle in rectangles:
                (x, y), h, w = capt_rectangle(rectangle)
                ratio = h / w
                if 2 > ratio > 0.5:
                    cv2.polylines(img, [np.array([[[x, y]], [[x, y + h]], [[x + w, y + h]], [[x + w, y]]])], True,
                                  (0, 255, 0), thickness=2)
            for circle in circles:
                (x, y), h, w = capt_rectangle(circle)
                ratio = h / w
                if 2 > ratio > 0.5:
                    cv2.polylines(img, [np.array([[[x, y]], [[x, y + h]], [[x + w, y + h]], [[x + w, y]]])], True,
                                  (0, 0, 255), thickness=2)
        i += 1
    pm.show_image(img, title + 'polygones')



def give_signs(img):
    """
    returns the array of the cropped_image containing the polygons
    :param img: image array
    :return: classification dictionnary of the images
    """
    image_masks = pm.each_image(pm.detect_image_component(img, detection_components), pm.image_contour)
    masks_polygons = pm.each_image(image_masks, find_shape, i=1)
    i = 0
    classified_polygons = {"triangles": {"rwb": [], "rw": []},
                        "rectangles": {"bw": []},
                        "circles": {"bw": [], "rwb": [], 'rw':[]}}
    for polygons in masks_polygons:
        triangles, rectangles, circles = interesting_polygon(polygons)
        """takes the three kinds of signs and the circle base color
        which is 0 for blue and 1 for red and classes them """
        for triangle in triangles:
            (x, y), h, w = capt_rectangle(triangle)
            crop_img = img[y:y + h, x:x + w]
            ratio = h / w
            translated_triangle = triangle - np.tile([[x, y]], (triangle.shape[0], 1, 1))
            clas = verif_triangle(translated_triangle, crop_img)
            if 2 > ratio > 0.5 and clas != "nothing":
                classified_polygons["triangles"][clas].append(crop_img)
        for rectangle in rectangles:
            (x, y), h, w = capt_rectangle(rectangle)
            crop_img = img[y:y + h, x:x + w]
            ratio = h / w
            translated_rectangle = rectangle - np.tile([[x, y]], (rectangle.shape[0], 1, 1))
            clas = verif_rectangle(translated_rectangle, crop_img)
            if 2 > ratio > 0.5 and clas != "nothing":
                classified_polygons["rectangles"][clas].append(crop_img)
        for circle in circles:
            (x, y), h, w = capt_rectangle(circle)
            crop_img = img[y:y + h, x:x + w]
            ratio = h / w
            translated_circle = circle - np.tile([[x, y]], (circle.shape[0], 1, 1))
            clas = verif_circle(translated_circle, i, crop_img)
            if 2 > ratio > 0.5 and clas != "nothing":
                classified_polygons["circles"][clas].append(crop_img)
        i+=1
    return classified_polygons
