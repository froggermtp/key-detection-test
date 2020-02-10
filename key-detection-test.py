import cv2
import imutils
import heapq
from collections import namedtuple
from pathlib import Path
import argparse


def setup_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--view_thresholds', action="store_true")
    return parser.parse_args()


def view_all_examples(examples):
    for example in examples:
        view_example(example.filename, example.threshold)


def view_example(filename, example):
    cv2.imshow(filename, example)
    cv2.waitKey(0)


def load_key(filename):
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    im = imutils.resize(im, width=300)
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

    return distances[0][1]


if __name__ == '__main__':
    args = setup_cli()
    training_set = setup_training_set()

    if args.view_thresholds:
        view_all_examples(training_set)

    keyA2 = load_key('keyA3.jpg')
    print(which_key(keyA2, training_set))
