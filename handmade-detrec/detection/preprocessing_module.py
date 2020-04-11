import cv2
import numpy as np

# CONSTANTS
COMPONENTS = {"blue": [[[95, 70, 50], [166, 255, 255]]],
              "red": [[[0, 50, 50], [14, 255, 255]], [[160, 100, 100], [179, 255, 255]]],
              "white": [[[0, 0, 100], [255, 100, 255]]], "black": [[[0, 0, 0], [255, 50, 60]]]}
SCALE_G = 0.5


def show_image(img, title, scale=SCALE_G):
    """
    Shows an image with the right window size
    :param img: numpy array with image components
    :param title: string with image window title
    :param scale: float specifying the visualisation scale
    :return:
    """
    newx, newy = int(img.shape[1] * scale), int(img.shape[0] * scale)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, (newx, newy))
    cv2.imshow(title, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def detect_image_component(image, components=COMPONENTS, show=False):
    """
    Shows the mask of each components on components list
    The component list has lists of 2 lists with 3 elements
    :param image: numpy array with image pixel values
    :param components: dictionnary specifying the color intervals for each color
    :param show: bool specifying if the image is shown
    :return: numpy array of the mask
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


def blur_image(image, show=False):
    """
    Blurs the image to have smoother contours
    :param image: numpy array with image pixel values
    :param show: bool specifying if the image is shown
    :return: blured immage array
    """
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    if show:
        show_image(blur, 'blur')
    return blur


def image_contour(image, show=False):
    """
    Returns the image edges after using a Gaussian Filter and the Canny edge detection technique
    :param image: numpy array with image pixel values
    :param show: bool specifying if the image is shown
    :return: array of contour image
    """
    """"""
    # application of the Gaussian filter
    blur = blur_image(image)
    # detection of the edges
    blur_copy = np.uint8(blur)
    edges = cv2.Canny(blur_copy, 100, 200)
    if show:
        show_image(edges, 'edges')
    return edges


def each_image(image_list, image_treatment, shows=False):
    """
    Applies image treatment on each image of image list
    :param image_list: list of numpy arrays
    :param image_treatment: function with show parameter
    :param shows: bool to show each image
    :return: list of treated numpy arrays
    """
    return list(map(lambda image: image_treatment(image, show=shows), image_list))


"""Tests"""
# Please try the functions with these lines

# show_image(cv2.imread(r'test_images/red_test_2.jpg'),'normal')
# detect_image_component(cv2.imread(r'test_images/red_test.jpg'),COMPONENTS, show = True)
# blur_image(detect_image_component(cv2.imread(r'test_images/red_test.jpg'),COMPONENTS)[0], show = True)
# image_contour(detect_image_component(cv2.imread(r'test_images/red_test.jpg'),COMPONENTS)[1], show = True)
# each_image(detect_image_component(cv2.imread(r'test_images/red_test.jpg'),COMPONENTS), blur_image, shows = True)
