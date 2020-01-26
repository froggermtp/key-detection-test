import cv2
import imutils
import heapq
from collections import namedtuple


def view_threshold(filename, thresh):
    cv2.imshow(filename, thresh)
    cv2.waitKey(0)


def load_key(filename):
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    im = imutils.resize(im, width=300)
    thresh = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 199, 5)
    Example = namedtuple('Example', 'filename threshold')

    #view_threshold(filename, thresh)

    return Example(filename, thresh)


def setup_training_set():
    filenames_for_training = ['keyA1.jpg', 'keyB.jpg']
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
    training_set = setup_training_set()
    keyA2 = load_key('keyA2.jpg')
    print(which_key(keyA2, training_set))
