import os
import re
from pathlib import Path
import cv2
import imutils
import numpy as np


# General Stuff


def find_jpgs_in_dir(path):
    return map(str, Path(path).glob('*.jpg'))


def extract_image_name(path):
    image_name = os.path.basename(os.path.normpath(path))
    regex = "^key(\w+)[0-9].jpg"
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
    grey = make_grey(img)
    blurred = do_gaussian_blur(grey)
    sobled = sobel(blurred)

    for x in range(len(sobled)):
        for y in range(len(sobled[x])):
            if sobled[x][y] < 20:
                sobled[x][y] = 0
            else:
                sobled[x][y] = 255

    contours = find_external_contours(sobled)
    largest_contour = get_largest_contour(contours)

    return largest_contour


def make_grey(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def do_gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 2, sigmaY=2)


def sobel(im):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    grad_x = cv2.Sobel(im, ddepth, 1, 0, ksize=3, scale=scale,
                       delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(im, ddepth, 0, 1, ksize=3, scale=scale,
                       delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad


def find_external_contours(img):
    mode = cv2.RETR_EXTERNAL
    method = cv2.CHAIN_APPROX_SIMPLE
    contours, _ = cv2.findContours(img, mode, method)
    return contours


def get_largest_contour(contours):
    return max(contours, key=cv2.contourArea)


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


if __name__ == "__main__":
    directory = "./keys/"
    paths = list(find_jpgs_in_dir(directory))
    images = [im for path, im in load_many_imgs(paths)]
    contours = [get_key_contour(im) for im in images]

    all_features = []

    for p, im, c in zip(paths, images, contours):
        # Uncomment these two lines to view the contours
        # view_contours(im, c)
        # pause()

        # All the features are combined into a single dictionary.
        # If you add a new feature, it needs to be added to this dictionary.
        features = {
            'class': extract_image_name(p),
            **hu_moments(c),
            **aspect_ratio(c),
            **extent(c),
            **solidity(c),
            **equivalent_diameter(c)
        }

        # I'm basically creating a space seperated file here.
        # This output will be piped into a file.
        all_features.append(features)
        print("id class feature val")

        for ii, features in enumerate(all_features):
            cls = features['class']

            for feature, val in features.items():
                if feature == 'class':
                    continue
                elif feature != 'class':
                    print("{} {} {} {}".format(ii, cls, feature, val))
