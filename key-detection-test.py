import cv2
import imutils
import numpy as np
import heapq
from collections import namedtuple
from pathlib import Path
import argparse


Example = namedtuple('Example', 'filename threshold')


def setup_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--view_thresholds', action="store_true")
    parser.add_argument('-i', '--ip', default='127.0.0.1')
    return parser.parse_args()


def view_all_examples(examples):
    for example in examples:
        view_example(example.filename, example.threshold)


def view_example(filename, example):
    cv2.imshow(filename, example)
    cv2.waitKey(0)


def load_key(filename):
    im = cv2.imread(filename)
    return process_image(filename, im)


def process_image(filename, im):
    resized = imutils.resize(im, 250)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # I'm messing around with different bluring techniques.
    # Don't know which is better. Most people use Gaussian.
    # The blur step reduces noise. I.E. makes the contours sharper.

    # im = cv2.medianBlur(im, 3)
    #blurred = cv2.bilateralFilter(im, 5, 75, 75)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobled = sobel(blurred)
    final_im = reduce_noise(sobled)

    contours = findContours(final_im)
    blank_image = np.zeros((600, 600, 3), np.uint8)
    cv2.drawContours(blank_image, contours, -1, (0, 255, 0), 3)

    # I hope that the contours for each key is unique enough to do the matching.
    # The contours can be plugged into the matchShape function. (Further down)
    # As you will see, the key on the wood background is not working well.
    # I don't know how to fix that. (Can we fix it???)

    return Example(filename, blank_image)


def sobel(im):
    # This is an algorithm for doing edge detection.
    # Someone should probably look into Canny Edge detection.
    # It's supposedly better...

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


def findContours(im):
    # Returns the largest contour from the image.
    # im -> do edge detection first before finding contours

    contours, hierarchy = cv2.findContours(
        im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # This is very naive. Will probably have to be changed.
    # Simply finds the largest contour. The key is unlikely to always be the largest contour.
    c = max(contours, key=cv2.contourArea)

    return c


def process_image2(filename, im):
    # Not currently being used.
    # I think I was messing with something. Don't remember.

    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 2, 1)

    return Example(filename, contours[0])


def setup_training_set():
    filenames_for_training = map(str, Path('./training-set').glob('*.jpg'))
    examples = [load_key(filename) for filename in filenames_for_training]

    return examples


def which_key(key, training_set):
    distances = []

    for example in training_set:
        d2 = cv2.matchShapes(
            key.threshold, example.threshold, cv2.CONTOURS_MATCH_I2, 0)
        heapq.heappush(distances, (d2, example.filename))

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

    # I have the video stuff disabled for now
    # open_camera(training_set, args.ip)
