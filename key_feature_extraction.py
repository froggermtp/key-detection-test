import os
import re
import math
from pathlib import Path
import random
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


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

def sample_contour(contour, k=300):
    perimeter = cv2.arcLength(contour, True)
    space = perimeter / k
    contour_array = sort_contour_points(contour)
    candidiates = np.empty((k, 2), dtype='int32')
    index = 1

    candidiates[0] = contour_array[0]

    for ii in range(1, len(contour_array)):
        if index == k:
            break

        x1, y1 = contour_array[ii]
        x2, y2 = candidiates[index - 1]

        dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

        if dist >= space:
            candidiates[index] = contour_array[ii]
            index += 1

    while index < k:
        candidiates[index] = random.choice(contour_array)
        index += 1

    new_contour = np.empty((k, 1, 2), dtype="int32")
    new_contour[:, 0, :] = candidiates

    return new_contour


def sample_contour_2(contour, k=64):
    perimeter = cv2.arcLength(contour, False)
    space = perimeter / k
    contour_array = sort_contour_points(contour)
    # contour_array = contour[:, 0, :]
    # candidiates = np.empty((k, 1, 2), dtype='int32')
    candidiates = []
    last_perimeter = 0

    candidiates.append(contour_array[0])

    for pnt in contour_array[1:]:
        potential = np.empty((len(candidiates) + 1, 1, 2), dtype="int32")
        for ii, candidiate in enumerate(candidiates):
            potential[ii] = candidiate
        potential[len(candidiates)] = pnt

        new_perimeter = cv2.arcLength(potential, False)

        if new_perimeter >= last_perimeter + space - .5:
            candidiates.append(pnt)
            last_perimeter = new_perimeter

    # while len(candidiates) < k:
    #     candidiates.append(random.choice(contour_array))

    potential = np.empty((len(candidiates), 1, 2), dtype="int32")
    for ii, candidiate in enumerate(candidiates):
        potential[ii] = candidiate
    return potential


def sample_contour_3(contour, k=64):
    indexes = list(range(len(contour)))
    random.shuffle(indexes)
    selected_indexes = sorted(indexes[:k])
    sampled_contour = contour[selected_indexes]
    return sampled_contour


def fourier_transform(contour):
    contour_array = contour[:, 0, :]
    contour_array = sort_contour_points(contour, contour_array)
    contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]
    fourier_result = np.fft.fft(contour_complex)
    fourier_result[0] = 0  # Translation invarience
    fourier_result /= abs(fourier_result[1])  # Scale invarience
    fourier_result = np.absolute(fourier_result)  # Rotation invarience
    return fourier_result[:400]


def fourier_transform_2(distances):
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
    # contour_array = contour[:, 0, :]
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
    response = list(range(len(imgs)))
    index = 0

    for img in imgs:
        contour = get_key_contour(img)
        center = find_center(contour)
        distances = find_distances(contour, center)
        distances = shift_distance(distances)
        fourier = fourier_transform_2(distances)[:20]
        fourier = shift_distance_max(list(fourier))
        samples[index] = fourier
        index += 1

    return samples, response


def shift_distance(dist):
    # max_index = dist.index(max(dist))
    # new_dist = dist[max_index:] + dist[:max_index]
    # return new_dist
    min_index = dist.index(min(dist))
    new_dist = dist[min_index:] + dist[:min_index]
    return new_dist


def shift_distance_max(dist):
    max_index = dist.index(max(dist))
    new_dist = dist[max_index:] + dist[:max_index]
    return new_dist


def plot_stuff(images, contours, filenames):
    fig = plt.figure(figsize=(8, 8))
    row = 1
    col = 2
    index = 1
    for image, contour, filename in zip(images, contours, filenames):
        fig.add_subplot(row, col, index)
        resized = imutils.resize(image, 500)
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(resized, (cX, cY), 7, (0, 255, 0), -1)
        with_contour = draw_contours_on_stencil(resized, contour)
        rgb = cv2.cvtColor(with_contour, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb)
        plt.axis("off")
        plt.title(filename)
        index += 1
    plt.show()


if __name__ == "__main__":
    # key1 = load_img("./test/KeyBravo01.jpg")
    # key2 = load_img("./train/KeyCharlie03.jpg")
    # contour1 = get_key_contour(key1)
    # center1 = find_center(contour1)
    # # contour1 = sample_contour_3(contour1, 700)
    # contour2 = get_key_contour(key2)
    # center2 = find_center(contour2)
    # # contour2 = sample_contour_3(contour2, 700)
    # dist1 = find_distances(contour1, center1)
    # dist1 = shift_distance(dist1)
    # dist2 = find_distances(contour2, center2)
    # dist2 = shift_distance(dist2)
    # fft1 = fourier_transform_2(dist1)[:70]
    # fft1 = shift_distance_max(list(fft1))
    # fft2 = fourier_transform_2(dist2)[:70]
    # fft2 = shift_distance_max(list(fft2))
    # plt.plot(fft1)
    # plt.plot(fft2)
    # plt.show()
    # plot_stuff([key1, key2], [contour1, contour2], ["One", "Two"])
    directory = "./test/"
    paths_train = list(find_jpgs_in_dir("./train/"))
    images_train = [im for im in load_many_imgs(paths_train)]
    samples_train, responses_train = sample_generator(images_train)
    responses_train = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    paths_test = list(find_jpgs_in_dir("./test/"))
    images_test = [im for im in load_many_imgs(paths_test)]
    samples_test, responses_test = sample_generator(images_test)

    K = 3
    # model = KNeighborsClassifier(n_neighbors=K)
    # model = svm.SVC()
    model = RandomForestClassifier()
    model.fit(samples_train, responses_train)

    print(model.predict(samples_test))

    # contours_train = [get_key_contour(im) for im in images_train]

    # for p, im, c in zip(paths, images, contours):
    #     ft = fourier_transform(c)

    # I was in the middle of implementing fourier shape descriptors
    # This is one of the main projects for next sprint.

    # all_features.append({
    #     'class': extract_image_name(p),
    #     'ft': ft
    # })

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

    # plot_stuff(images_train, c, paths_train)
