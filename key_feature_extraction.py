import os
import re
from pathlib import Path
import cv2
import imutils
import numpy as np
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
import matplotlib.pyplot as plt


# General Stuff


def find_jpgs_in_dir(path):
    return map(str, Path(path).glob('*.jpg'))


def extract_image_name(path):
    image_name = os.path.basename(os.path.normpath(path))
    regex = "^(?i)key([a-zA-Z]+)[0-9]+.jpg"
    return re.split(regex, image_name)[1]


def load_many_imgs(lst_of_paths):
    return [(path, load_img(path)) for path in lst_of_paths]


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
# All of these functions take in contours and return a dictionary entry.
# We can also calculate masks, histograms, etc but I haven't found a good reason to yet.
# 4/02/20 update: Based on my tests, none of these basic features are going to work.
# I've instead pivoted to look at fourier shape descriptors.


def hu_moments(contour):
    moments = cv2.HuMoments(cv2.moments(contour)).flatten()
    moments_dict = {}

    for ii, m in enumerate(moments):
        key = "hu-{}".format(ii)
        moments_dict[key] = m

    return moments_dict


def aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w)/h
    return {'aspect_ratio': aspect_ratio}


def extent(contour):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w*h
    extent = float(area)/rect_area
    return {'extent': extent}


def solidity(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    return {'solidity': solidity}


def equivalent_diameter(contour):
    area = cv2.contourArea(contour)
    equi_diameter = np.sqrt(4*area/np.pi)
    return {'equivalent_diameter': equi_diameter}


def fourier_transform(contour):
    contour_array = contour[:, 0, :]
    contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]
    fourier_result = np.fft.fft(contour_complex)
    return fourier_result


def plot_stuff(images, contours, filenames):
    fig = plt.figure(figsize=(8, 8))
    row = 4
    col = 4
    index = 1
    for image, contour, filename in zip(images, contours, filenames):
        fig.add_subplot(row, col, index)
        resized = imutils.resize(image, 500)
        with_contour = draw_contours_on_stencil(resized, contour)
        rgb = cv2.cvtColor(with_contour, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb)
        plt.axis("off")
        plt.title(filename)
        index += 1
    plt.show()


if __name__ == "__main__":
    directory = "./keys/"
    paths = list(find_jpgs_in_dir(directory))
    images = [im for path, im in load_many_imgs(paths)]
    contours = [get_key_contour(im) for im in images]

    all_features = []

    for p, im, c in zip(paths, images, contours):
        ft = fourier_transform(c)

        # I was in the middle of implementing fourier shape descriptors
        # This is one of the main projects for next sprint.

        all_features.append({
            'class': extract_image_name(p),
            'ft': ft
        })

        # All the features are combined into a single dictionary.
        # If you add a new feature, it needs to be added to this dictionary.
        # features = {
        #     'class': extract_image_name(p),
        #     **hu_moments(c),
        #     **aspect_ratio(c),
        #     **extent(c),
        #     **solidity(c),
        #     **equivalent_diameter(c)
        # }

        # I'm basically creating a space seperated file here.
        # This output will be piped into a file.
        # all_features.append(features)
        # print("id class feature val")

        # for ii, features in enumerate(all_features):
        #     cls = features['class']

        #     for feature, val in features.items():
        #         if feature == 'class':
        #             continue
        #         elif feature != 'class':
        #             print("{} {} {} {}".format(ii, cls, feature, val))

        # Fourier stuff
        # for ii, features in enumerate(all_features):
        #     cls = features['class']
        #     ft = features['ft']

        #     for jj, ft_val in enumerate(ft):
        #         print("{} {} ft{} {}".format(ii, cls, jj, ft_val))

    plot_stuff(images, contours, paths)
