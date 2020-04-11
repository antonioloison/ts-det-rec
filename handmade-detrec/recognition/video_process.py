"""
Run video to try the neural networks
"""

from neural_network import *
from interface import *
import cv2

IMAGE_SIZE = 28 * 28 * 2

cam = cv2.VideoCapture(0)
NN_blue_circles = Nnetwork([IMAGE_SIZE, 8, 7, 3], [sigmoid, sigmoid, sigmoid, sigmoid])
NN_rectangles = Nnetwork([IMAGE_SIZE, 6, 3], [sigmoid, sigmoid, sigmoid, sigmoid])
NN_red_circles = Nnetwork([IMAGE_SIZE, 6, 4], [sigmoid, sigmoid, sigmoid, sigmoid])
NN_red_triangles = Nnetwork([IMAGE_SIZE, 6, 3], [sigmoid, sigmoid, sigmoid, sigmoid])
NN_blue_circles.load(
    r"neural_networks/saved_weightsblue_circles.npy",
    r"neural_networks/saved_biasblue_circles.npy")
NN_rectangles.load(
    r"neural_networks/saved_weightsrectangles.npy",
    r"neural_networks/saved_biasrectangles.npy")
NN_red_circles.load(
    r"neural_networks/saved_weightsred_circles.npy",
    r"neural_networks/saved_biasred_circles.npy")
NN_red_triangles.load(
    r"neural_networks/saved_weightsred_triangles.npy",
    r"neural_networks/saved_biasred_triangles.npy")

neural_networks = {"trianglesred": NN_red_triangles, "rectanglesblue": NN_rectangles,
                   "circlesred": NN_red_circles, "circlesblue": NN_blue_circles}

while True:
    ret, img = cam.read()
    img = cv2.resize(img, (1060, int(220 / 340 * 1060)))

    detected_signs(img, neural_networks)
    cv2.imshow("cam", img)
    cv2.waitKey(10)
