from matplotlib import pyplot as plt
from skimage.feature import hog
from matplotlib import patches
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib import ListedColormap
import numpy as np
# from pull_digit import labeled_user_image
import cv2

data_training = cv2.imread('')