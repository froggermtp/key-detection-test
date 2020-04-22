import key_feature_extraction as key
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score


def load_images(path):
    img_paths = list(key.find_jpgs_in_dir(path))
    images = key.load_many_imgs(img_paths)

    return (img_paths, images)


def create_mapping(img_names):
    mapping = {name: ii for ii, name in enumerate(set(img_names))}

    return mapping


def create_labels(img_names, mapping):
    labels = [mapping[name] for name in img_names]

    return labels


def test_models(models, train_path="./train", test_path="./test"):
    train_path, train_images = load_images(train_path)
    test_path, test_images = load_images(test_path)
    train_names = [key.extract_image_name(p) for p in train_path]
    test_names = [key.extract_image_name(p) for p in test_path]
    mapping = create_mapping(train_names + test_names)
    train_labels = create_labels(train_names, mapping)
    test_labels = create_labels(test_names, mapping)
    train_samples = key.sample_generator(train_images)
    test_samples = key.sample_generator(test_images)

    for model in models:
        model.fit(train_samples, train_labels)
        results = model.predict(test_samples)
        score = accuracy_score(test_labels, results)
        print(model.__class__)
        print(score)


if __name__ == "__main__":
    models = [svm.SVC(class_weight='balanced'),
              RandomForestClassifier(),
              KNeighborsClassifier(n_neighbors=1),
              GaussianNB(),
              tree.DecisionTreeClassifier()]
    test_models(models)
    print('done')
