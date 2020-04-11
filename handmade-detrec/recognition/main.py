"""
TO DO
Main file 
"""

from neural_network import *
from interface import *

IMAGE_SIZE = 28*28*2

image = cv2.imread(r"../detection/test_images/stop.jpg")

NN_blue_circles = Nnetwork([IMAGE_SIZE, 8, 7, 3], [sigmoid, sigmoid, sigmoid, sigmoid])
NN_rectangles = Nnetwork([IMAGE_SIZE, 6, 3], [sigmoid, sigmoid, sigmoid, sigmoid])
NN_red_circles = Nnetwork([IMAGE_SIZE, 6, 4], [sigmoid, sigmoid, sigmoid, sigmoid])
NN_red_triangles = Nnetwork([IMAGE_SIZE, 6, 3], [sigmoid, sigmoid, sigmoid, sigmoid])


NN_blue_circles.load(r"neural_networks/saved_weightsblue_circles.npy",
                     r"neural_networks/saved_biasblue_circles.npy")
NN_rectangles.load(r"neural_networks/saved_weightsrectangles.npy",
                   r"neural_networks/saved_biasrectangles.npy")
NN_red_circles.load(r"neural_networks/saved_weightsred_circles.npy",
                    r"neural_networks/saved_biasred_circles.npy")
NN_red_triangles.load(r"neural_networks/saved_weightsred_triangles.npy",
                      r"neural_networks/saved_biasred_triangles.npy")


neural_networks = {"trianglesred": NN_red_triangles, "rectanglesblue": NN_rectangles,
                   "circlesred": NN_red_circles, "circlesblue": NN_blue_circles}
