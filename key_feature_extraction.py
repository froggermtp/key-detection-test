import os
import re
import math
from pathlib import Path
import cv2
import imutils
import numpy as np


# General Stuff


def find_jpgs_in_dir(path):
    return map(str, Path(path).glob('*.jpg'))


def extract_image_name(path):
    image_name = os.path.basename(os.path.normpath(path))
    regex = "^(?i)key([a-zA-Z]+)[0-9]+.jpg"
    return re.split(regex, image_name)[1]


def load_many_imgs(lst_of_paths):
    return [load_img(path) for path in lst_of_paths]


def load_img(path):
    return cv2.imread(path)


def show_img(img, title="Test"):
    cv2.imshow(title, resize_img(img, 500))


def pause():
    cv2.waitKey(0)


def resize_img(img, size=500):
    return imutils.resize(img, size)


# Image Processing


def get_key_contour(img):
    prepped = preprocess_img(img)
    contours = find_external_contours(prepped)
    largest_contour = get_largest_contour(contours)

    return largest_contour


def preprocess_img(img):
    gridsize = 1

    bgr = imutils.resize(img, 500)

    # Use CLAHE to normalize light
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.9, tileGridSize=(gridsize, gridsize))
    lab[..., 0] = clahe.apply(lab[..., 0])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    filtered = cv2.bilateralFilter(bgr, 9, 10000, 10000)
    edges = cv2.Canny(filtered, 255, 255, apertureSize=7, L2gradient=True)

    return edges


def find_external_contours(img):
    mode = cv2.RETR_EXTERNAL
    method = cv2.CHAIN_APPROX_NONE
    contours, _ = cv2.findContours(img, mode, method)
    return contours


def get_largest_contour(contours):
    def f(x): return cv2.arcLength(x, True)
    return max(contours, key=f)


def view_contours(img, contours):
    show_img(
        draw_contours_on_stencil(
            img,
            contours
        )
    )


def get_contours_img(img, contours):
    stencil = create_stencil(img)
    return draw_contours_on_stencil(stencil, contours)


def create_stencil(img):
    return np.zeros(img.shape, np.uint8)


def draw_contours_on_stencil(stencil, contours):
    cv2.drawContours(stencil, contours, -1, (0, 255, 0), 5)
    return stencil


# Feature Extraction

def fourier_transform(distances):
    fourier_result = np.fft.fft(distances)
    fourier_result = fourier_result[:len(fourier_result) // 2]
    fourier_result = np.abs(fourier_result)
    # fourier_result /= fourier_result[1]
    fourier_result = np.divide(fourier_result, fourier_result[0])
    for ii, x in enumerate(fourier_result):
        if np.isnan(fourier_result[ii]):
            fourier_result[ii] = 0
    return fourier_result[1:]


def find_distances(contour, center):
    cX, cY = center
    contour_array = sort_contour_points(contour)
    distances = []

    for x, y in contour_array:
        dist = math.sqrt((x - cX)**2 + (y - cY)**2)
        distances.append(dist)

    return distances


def find_center(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY


def sort_contour_points(contour):
    contour_array = contour[:, 0, :]
    thetas = []
    cX, cY = find_center(contour)

    for x, y in contour_array:
        theta = math.atan2(y - cY, x - cX)
        thetas.append(theta)

    to_sort = [(theta, arr) for theta, arr in zip(thetas, contour_array)]
    was_sorted = sorted(to_sort, key=lambda x: x[0])
    return np.array([tup[1] for tup in was_sorted])


def sample_generator(imgs):
    samples = np.empty((len(imgs), 20), dtype="float64")
    index = 0

    for img in imgs:
        contour = get_key_contour(img)
        center = find_center(contour)
        distances = find_distances(contour, center)
        distances = shift_distance_min(distances)
        fourier = fourier_transform(distances)[:20]
        fourier = shift_distance_max(list(fourier))
        samples[index] = fourier
        index += 1

    return samples


def shift_distance_min(dist):
    min_index = dist.index(min(dist))
    new_dist = dist[min_index:] + dist[:min_index]
    return new_dist


def shift_distance_max(dist):
    max_index = dist.index(max(dist))
    new_dist = dist[max_index:] + dist[:max_index]
    return new_dist
