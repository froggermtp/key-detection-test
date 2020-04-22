import argparse
import cv2
import matplotlib.pyplot as plt
import key_feature_extraction as key


def setup_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("-c", "--canny", action="store_true")
    parser.add_argument("-d", "--distance")
    parser.add_argument("-f", "--fourier")
    parser.add_argument("--centroid_off", action="store_false")

    return parser.parse_args()


def view_single_key(img_path, centroid=True):
    image = key.load_img(img_path)
    contour = key.get_key_contour(image)
    image = key.resize_img(image)

    if centroid:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(image, (cX, cY), 7, (0, 255, 0), -1)

    key.view_contours(image, contour)
    key.pause()


def view_single_key_canny(img_path):
    image = key.load_img(img_path)
    canny = key.preprocess_img(image)
    key.show_img(canny)
    key.pause()


def compare_dist(img_path_1, img_path_2):
    key1 = key.load_img(img_path_1)
    key2 = key.load_img(img_path_2)

    contour1 = key.get_key_contour(key1)
    center1 = key.find_center(contour1)
    contour2 = key.get_key_contour(key2)
    center2 = key.find_center(contour2)

    dist1 = key.find_distances(contour1, center1)
    dist1 = key.shift_distance_min(dist1)
    dist2 = key.find_distances(contour2, center2)
    dist2 = key.shift_distance_min(dist2)

    plt.plot(dist1)
    plt.plot(dist2)
    plt.show()


def compare_fourier(img_path_1, img_path_2):
    key1 = key.load_img(img_path_1)
    key2 = key.load_img(img_path_2)

    contour1 = key.get_key_contour(key1)
    center1 = key.find_center(contour1)
    contour2 = key.get_key_contour(key2)
    center2 = key.find_center(contour2)

    dist1 = key.find_distances(contour1, center1)
    dist1 = key.shift_distance_min(dist1)
    dist2 = key.find_distances(contour2, center2)
    dist2 = key.shift_distance_min(dist2)

    fft1 = key.fourier_transform(dist1)[:70]
    fft1 = key.shift_distance_max(list(fft1))
    fft2 = key.fourier_transform(dist2)[:70]
    fft2 = key.shift_distance_max(list(fft2))

    plt.plot(fft1)
    plt.plot(fft2)
    plt.show()


if __name__ == "__main__":
    args = setup_cli()

    if args.distance:
        compare_dist(args.path, args.distance)
    elif args.fourier:
        compare_fourier(args.path, args.fourier)
    elif args.canny:
        view_single_key_canny(args.path)
    else:
        view_single_key(args.path, args.centroid_off)
