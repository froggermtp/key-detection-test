import cv2
import imutils
import numpy as np
import heapq
from collections import namedtuple
from pathlib import Path
import argparse
import math


Example = namedtuple('Example', 'filename threshold features')


def setup_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--view_thresholds', action="store_true")
    parser.add_argument('-v', '--video', action="store_true")
    parser.add_argument('-i', '--ip', default='127.0.0.1')
    return parser.parse_args()


def view_all_examples(examples):
    for example in examples:
        view_example(example.filename, example.threshold)

    cv2.waitKey(0)


def view_example(filename, example):
    cv2.imshow(filename, example)


def load_key(filename):
    im = cv2.imread(filename)
    return process_image(filename, im)


def process_image(filename, im):
    resized = imutils.resize(im, 250)
    im_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = canny_edge_detection(im_gray)
    final_im = reduce_noise(edges)
    contours = findContours(final_im)
    features = extract_features(im_gray, resized, contours)

    blank_image = np.zeros(resized.shape, np.uint8)
    cv2.drawContours(blank_image, contours, -1, (0, 255, 0), 3)

    return Example(filename, blank_image, features)


def extract_features(gray_im, im, contours):
    mask = compute_mask(gray_im, contours)
    mean_val = compute_mean_contour_value(im, mask)
    aspect_ratio = compute_aspect_ratio(contours)
    extent = compute_extent(contours)
    return [*mean_val, aspect_ratio, extent]


def compute_mask(gray_im, contours):
    mask = np.zeros(gray_im.shape, np.uint8)
    cv2.drawContours(mask, [contours], 0, 255, -1)
    return mask


def compute_mean_contour_value(im, mask):
    mean_val = cv2.mean(im, mask=mask)
    return mean_val


def compute_aspect_ratio(contours):
    x, y, w, h = cv2.boundingRect(contours)
    aspect_ratio = float(w) / h
    return aspect_ratio


def compute_extent(contours):
    area = cv2.contourArea(contours)
    x, y, w, h = cv2.boundingRect(contours)
    rect_area = w * h
    extent = float(area) / rect_area
    return extent


def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - \
        ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - \
        ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def canny_edge_detection(im_gray):
    im_gray_blurred = cv2.GaussianBlur(im_gray, (5, 5), 0)
    edges = cv2.Canny(im_gray, 0, 100)

    return edges


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


def reduce_noise(im):
    mean = np.mean(im)
    im[im <= mean] = 0

    return im


def findContours(edges):
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # This is very naive. Will probably have to be changed.
    # Simply finds the largest contour. The key is unlikely to always be the largest contour.
    c = max(contours, key=cv2.contourArea)

    return contours


def setup_training_set():
    filenames_for_training = map(str, Path('./training-set').glob('*.jpg'))
    examples = [load_key(filename) for filename in filenames_for_training]

    return examples


def setup_test_set():
    filenames_for_testing = map(str, Path('./test-set').glob('*.jpg'))
    examples = [load_key(filename) for filename in filenames_for_testing]

    return examples


def run_trials(training_set, test_set):
    for img in test_set:
        classification = which_key(img, training_set)
        msg = 'Classify {0} as {1}'.format(img.filename, classification)
        print(msg)


def which_key(key, training_set):
    distances = []

    for example in training_set:
        # print(example.filename)
        # print(key.features)
        # print(example.features)
        d2 = cv2.matchShapes(
            key.threshold, example.threshold, cv2.CONTOURS_MATCH_I2, 0)

        distance = 0
        for ii in range(len(example.features)):
            term1 = key.features[ii] - example.features[ii]
            term2 = term1**2
            distance += term2

        distance = math.sqrt(distance) + d2

        heapq.heappush(distances, (distance, example.filename))

    print(distances)

    return distances[0][1]


def open_camera(training_set, ip):
    URL = "http://" + ip + ":8080/video"

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(URL)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        example = process_image('camera', frame)
        which_key(example, training_set)
        frame = example.threshold

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow("preview")
    vc.release()


if __name__ == '__main__':
    args = setup_cli()
    training_set = setup_training_set()

    if args.view_thresholds:
        view_all_examples(training_set)

    if not args.video:
        test_set = setup_test_set()
        view_all_examples(test_set)
        run_trials(training_set, test_set)
    else:
        open_camera(training_set, args.ip)
