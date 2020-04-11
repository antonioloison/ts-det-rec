import os
import preprocessing_module as pm
import math
import numpy as np
import cv2

# CONSTANTS
# create function to have better components
DETECTION_COMPONENTS = {"blue": [[[95, 70, 50], [166, 255, 255]]],
                        "red": [[[0, 50, 50], [14, 255, 255]], [[160, 100, 100], [179, 255, 255]]]}
HELP_COMPONENTS = {"white": [[[0, 0, 100], [255, 100, 255]]], "black": [[[0, 0, 0], [255, 80, 60]]]}


def capt_rectangle(arr):
    """
    Returns the points of the biggest rectangle containing the array
    :param arr: numpy array
    :return: tuple with left bottom corner coordinates and the width and height of the rectangle
    """
    left_abs = math.inf
    right_abs = 0
    bottom_ord = math.inf
    top_ord = 0
    n = arr.shape[0]
    for i in range(n):
        point = arr[i, 0]
        absi, ordin = point[0], point[1]
        if left_abs > absi:
            left_abs = absi
        if right_abs < absi:
            right_abs = absi
        if bottom_ord > ordin:
            bottom_ord = ordin
        if top_ord < ordin:
            top_ord = ordin
    origin = left_abs, bottom_ord
    height = top_ord - bottom_ord
    width = right_abs - left_abs
    return origin, height, width


def find_shape(image, show=False):
    """
    Returns the arrays of the polygons in the image
    showing the image with the polygons if necessary
    :param image: image array
    :param show: bool to specify if the image must be shown
    :return:
    """
    edges = pm.image_contour(image)
    sh = image.shape
    canvas = np.zeros((sh[0], sh[1]))
    contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    polygons = []
    for contour in contours:
        epsilon = 0.03 * cv2.arcLength(contour, True)  # calculates the epsilon from first contour length
        polyg = cv2.approxPolyDP(contour, epsilon, True)
        polygons.append(polyg)

    polyg_img = cv2.drawContours(canvas, polygons, -1, (255, 255, 255), 1)

    if show:
        pm.show_image(polyg_img, 'polyg_imag')

    return polygons


def is_narrow(polygon):
    s = polygon.shape[0]
    maxi = 0
    mini = math.inf
    for x in range(s):
        long = cv2.arcLength(np.array([[polygon[x, 0]], [polygon[(x + 1) % s, 0]]]), False)
        if long > maxi:
            maxi = long
        if long < mini:
            mini = long
    return maxi > 2 * mini


def is_contained(polygon1, polygon2):
    """
    Checks if every point of polygon1 is inside polygon2
    :param polygon1: numpy array with polygon point coordinates
    :param polygon2: numpy array with polygon point coordinates
    :return:
    """
    s = polygon1.shape
    if cv2.contourArea(polygon1) < 0.1 * cv2.contourArea(polygon2):
        return False
    for x in range(s[0]):
        if cv2.pointPolygonTest(polygon2, (polygon1[x, 0, 0], polygon1[x, 0, 1]), False) < 0:
            return False
    return True


def principal_polygons(polygons):
    """
    Filters the main polygons detected
    :param polygons: list of polygons
    :return: filtered polygons
    TO FIX THIS FUNCTION IS WEIRD
    """
    plg = polygons.copy()
    n = len(polygons)
    for i in range(n):
        for j in range(n):
            if is_contained(polygons[j], polygons[i]) and i != j:
                plg = [x for x in plg if not np.array_equal(x, polygons[j])]
    return plg


def interesting_polygon(polygons):
    """
    Finds the interesting polygons that have a large enough shape
    in the list of polygons and sorts them out
    :param polygons: list of polygons
    :return: tuple of lists of interesting trinangles, rectangles and circles
    """
    triangles = []
    rectangles = []
    circles = []
    polyg = []
    for polygon in polygons:
        point_nbre = len(polygon)

        if point_nbre >= 3 and cv2.isContourConvex(polygon) \
                and (cv2.contourArea(polygon) > 1200 or cv2.arcLength(polygon, True) > 500) \
                and (cv2.contourArea(polygon) < 10000 or cv2.arcLength(polygon, True) < 1500):
            # if point_nbre >= 8:
            polyg.append(polygon)
    for polygon in principal_polygons(polyg):
        point_nbre = len(polygon)
        if point_nbre > 6:
            circles.append(polygon)
        elif not (is_narrow(polygon)):
            if point_nbre == 3:
                triangles.append(polygon)
            elif point_nbre == 4:
                rectangles.append(polygon)
    return triangles, rectangles, circles


def sign_finder(image):
    """
    Shows the cropped image containing the traffic sign
    :param image: big image possibly containing traffic signs
    :return: None
    """
    img = cv2.imread(image)
    image_masks = pm.each_image(pm.detect_image_component(img, DETECTION_COMPONENTS), pm.image_contour)
    masks_polygons = pm.each_image(image_masks, find_shape, i=1)
    for polygons in masks_polygons:
        triangles, rectangles, circles = interesting_polygon(polygons)
        for triangle in triangles:
            (x, y), h, w = capt_rectangle(triangle)
            crop_image = img[y:y + h, x:x + w]
            cv2.imshow("triangle", crop_image)
            cv2.waitKey()
            cv2.destroyAllWindows()
        for rectangle in rectangles:
            (x, y), h, w = capt_rectangle(rectangle)
            crop_image = img[y:y + h, x:x + w]
            cv2.imshow("rectangle", crop_image)
            cv2.waitKey()
            cv2.destroyAllWindows()
        for circle in circles:
            (x, y), h, w = capt_rectangle(circle)
            crop_image = img[y:y + h, x:x + w]
            cv2.imshow("circle", crop_image)
            cv2.waitKey()
            cv2.destroyAllWindows()


def polygon_shower(image_path):
    """
    Shows the colored polygons on the image
    :param image_path: Path of the image
    :return: None
    """
    img = cv2.imread(image_path)
    image_masks = pm.each_image(pm.detect_image_component(img, DETECTION_COMPONENTS), pm.image_contour)
    masks_polygons = pm.each_image(image_masks, find_shape, i=1)
    for polygons in masks_polygons:
        triangles, rectangles, circles = interesting_polygon(polygons)
        cv2.polylines(img, triangles, True, (0, 255, 0), thickness=2)
        cv2.polylines(img, rectangles, True, (255, 0, 0), thickness=2)
        cv2.polylines(img, circles, True, (0, 0, 255), thickness=2)
    pm.show_image(img, 'polygones')


'''Classification of the polygons'''

'''Simple classification'''


def easy_give_signs(img):
    """
    Classifies the polygons in image according to their shape and color
    :param img: image array
    :return: polygon classification
    """
    image_masks = pm.each_image(pm.detect_image_component(img, DETECTION_COMPONENTS), pm.image_contour)
    masks_polygons = pm.each_image(image_masks, find_shape, i=1)
    i = 0
    classified_polygons = {"triangles": {"red": [], "blue": []},
                           "rectangles": {"red": [], "blue": []},
                           "circles": {"red": [], "blue": []}}
    for polygons in masks_polygons:
        triangles, rectangles, circles = interesting_polygon(polygons)
        """takes the three kinds of signs and the circle base color
        which is 0 for blue and 1 for red and classes them """
        for triangle in triangles:
            (x, y), h, w = capt_rectangle(triangle)
            crop_img = img[y:y + h, x:x + w]
            ratio = h / w
            if 2 > ratio > 0.5:
                if i:
                    classified_polygons["triangles"]["red"].append((((x, y, h, w)), crop_img))
                else:
                    classified_polygons["triangles"]["blue"].append((((x, y, h, w)), crop_img))
        for rectangle in rectangles:
            (x, y), h, w = capt_rectangle(rectangle)
            crop_img = img[y:y + h, x:x + w]
            ratio = h / w
            if 2 > ratio > 0.5:
                if i:
                    classified_polygons["rectangles"]["red"].append(((x, y, h, w), crop_img))
                else:
                    classified_polygons["rectangles"]["blue"].append(((x, y, h, w), crop_img))
        for circle in circles:
            (x, y), h, w = capt_rectangle(circle)
            crop_img = img[y:y + h, x:x + w]
            ratio = h / w
            if 2 > ratio > 0.5:
                if i:
                    classified_polygons["circles"]["red"].append(((x, y, h, w), crop_img))
                else:
                    classified_polygons["circles"]["blue"].append(((x, y, h, w), crop_img))
        i += 1
    return classified_polygons


def show_polygones(dico):
    """test function to see if the precedent function returns the good arrays"""
    for key in dico.keys():
        for sub_key in dico[key].keys():
            for polygon in dico[key][sub_key]:
                pm.show_image(polygon, key + sub_key)


def detect_directory(directory, function):
    """takes a directory path containing the images and detects the signs
    in each image in the directory"""
    for root, dirs, files in os.walk(directory):
        for filename in files:
            function(directory + '\\' + filename, filename)


"""Tests"""
"""Feel free to test the functions on the test images"""
# sign_finder(r'test_images/red_test.jpg')  # works
# sign_finder('test_images/red_test_2.jpg')
# sign_finder('test_images/stop.jpg') #works
# sign_finder('test_images/russian_image.jpg') #works
# sign_finder('test_images/test.jpg') # works more or less

# polygon_shower(r'test_images/red_test.jpg')
# polygon_shower('test_images/red_test_2.jpg')# works
# polygon_shower('test_images/stop.jpg') # works
# polygon_shower('test_images/russian_image.jpg') # works
# polygon_shower('test_images/test.jpg') # works more or less
# polygon_shower('test_images/speed_limit_test.jpg')# works

# print(give_signs(cv2.imread(r'test_images/red_test.jpg')))
