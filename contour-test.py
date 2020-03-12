import cv2
import imutils
import numpy as np
from functools import reduce
from pathlib import Path


def compose(*fs):
    return reduce(compose2, fs)


def compose2(f, g):
    return lambda *a, **kw: f(g(*a, **kw))


def contour_test(path):
    img = white_balance(load_img(path))
    grey = make_grey(img)

    #thresh = close_holes(remove_noise(do_threshold(grey)))
    #thresh = remove_noise(do_threshold(grey))
    #thresh = cv2.bitwise_not(thresh)
    #grey = do_gaussian_blur(grey)
    grey = cv2.GaussianBlur(grey, (5, 5), 2, sigmaY=2)
    thresh = sobel(grey)
    thresh = do_threshold(thresh)
    #thresh = cv2.dilate(thresh, None, iterations=6)
    c = get_largest_contour(find_external_contours(thresh))
    sure_fg, unknown = get_forground_and_background(thresh)
    markers = get_markers(sure_fg, unknown)
    watershed_img = make_grey(do_watershed(img, markers))
    contours = find_external_contours(watershed_img)
    largest_contour = get_largest_contour(contours)
    show_img(watershed_img, "bar")
    view_contours(img, largest_contour)


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


def load_img(path):
    return cv2.imread(path)


def make_grey(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def resize_img(img, size=500):
    return imutils.resize(img, size)


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


def find_edges(img):
    return do_canny_edge_detection(
        do_gaussian_blur(img)
    )


def do_gaussian_blur(img):
    # 35 35
    return cv2.GaussianBlur(img, (21, 21), 0)


def do_canny_edge_detection(img):
    #blur = do_gaussian_blur(img)
    return cv2.Canny(img, 0, 100)


def do_threshold(img):
    #blur = do_gaussian_blur(img)
    # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                cv2.THRESH_BINARY_INV, 7, 1.5)
    ret_otsu, im_bw_otsu = cv2.threshold(
        img, 80, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return im_bw_otsu


def close_holes(thresh):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)


def remove_noise(thresh):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, None, iterations=0)


def get_forground_and_background(thresh):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    # sure background area
    sure_bg = cv2.dilate(thresh, None, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 0)
    ret, sure_fg = cv2.threshold(
        dist_transform, 0.1*dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    #sure_fg = cv2.erode(thresh, None, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    return (sure_fg, unknown)


def get_markers(sure_fg, unknown):
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    return markers


def do_watershed(img, markers):
    watershed_markers = cv2.watershed(img, markers)
    stencil = create_stencil(img)
    stencil[watershed_markers == -1] = [255, 255, 255]
    for row, _ in enumerate(stencil):
        for col, _ in enumerate(stencil[row]):
            if row == stencil.shape[0] - 1:
                stencil[row][col] = 0
            elif col == stencil.shape[1] - 1:
                stencil[row][col] = 0
            elif row == 0 or col == 0:
                stencil[row][col] = 0
    return stencil


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


def create_stencil(img):
    return np.zeros(img.shape, np.uint8)


def draw_contours_on_stencil(stencil, contours):
    cv2.drawContours(stencil, contours, -1, (0, 255, 0), 5)
    return stencil


def brisk_test():
    extract = setup_brisk()
    training_set = [(name, extract(img)) for name, img in setup_training_set()]
    test_set = [(name, extract(img)) for name, img in setup_test_set()]
    match = setup_matcher()

    for test_name, test_des in test_set:
        distances = []

        for training_name, training_des in training_set:
            m = match(test_des, training_des)
            d = sum([dmatch.distance for lst in m for dmatch in lst])
            distances.append((training_name, d))

        dist_sorted = sorted(distances, key=lambda x: x[1])

        print(dist_sorted, "\n")
        print("Matched {} with {}".format(test_name, dist_sorted[0][0]), "\n")


def setup_training_set():
    path = './training-set'
    return load_many_imgs(
        find_jpgs_in_dir(path)
    )


def setup_test_set():
    path = './test-set'
    return load_many_imgs(
        find_jpgs_in_dir(path)
    )


def load_many_imgs(lst_of_paths):
    return [(path, load_img(path)) for path in lst_of_paths]


def find_jpgs_in_dir(path):
    return map(str, Path(path).glob('*.jpg'))


def setup_brisk():
    # brisk = cv2.BRISK_create(60, 4, 1)
    orb = cv2.ORB_create()
    return lambda img: extract_description(orb, img)


def extract_description(feature_extractor, img):
    kp, des = feature_extractor.detectAndCompute(img, None)
    return des


def setup_matcher():
    matcher = cv2.BFMatcher()
    return lambda des1, des2: get_matches(matcher, des1, des2)


def get_matches(matcher, des1, des2):
    return matcher.knnMatch(des1, des2, k=2)


def apply_ratio_test(matches):
    good = []

    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    return good


def show_img(img, title="Test"):
    cv2.imshow(title, resize_img(img, 500))


def pause():
    cv2.waitKey(0)


if __name__ == "__main__":
    contour_test("./test-set/keyB2.jpg")
    # brisk_test()
    # img = load_img("./test-set/keyA2.jpg")
    # img = resize_img(img, 500)
    # gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # blur = cv2.GaussianBlur(gs, (25, 25), 0)
    # ret_otsu, im_bw_otsu = cv2.threshold(
    #     blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # cv2.imshow("foo", img)
    # cv2.imshow("test", im_bw_otsu)
    # cv2.imshow("test", resize_img(find_edges(img), 500))
    pause()
