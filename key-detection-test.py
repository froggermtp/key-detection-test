import cv2
import imutils
import heapq
from collections import namedtuple
from pathlib import Path
import argparse


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
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # im = imutils.resize(im, width=300)
    thresh = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 199, 5)
    Example = namedtuple('Example', 'filename threshold')

    return Example(filename, thresh)


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

    open_camera(training_set, args.ip)
