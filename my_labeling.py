_authors_ = 'TO_BE_FILLED'
_group_ = 'TO_BE_FILLED'

import numpy as np
from utils_data import read_dataset, visualize_retrieval
import Kmeans as km
import KNN as kn

def Retrieval_by_color(image_list, label_list, col):
    col_list = [i for i, et in zip(image_list, label_list) if col in et]
    col_list = np.array(col_list)
    visualize_retrieval(col_list, 10)


def Retrieval_by_shape(image_list, label_list, sha):
    sha_list = [i for i, et in zip(image_list, label_list) if et == sha]
    sha_list = np.array(sha_list)
    visualize_retrieval(sha_list, 10)

def Retrieval_combined(image_list, label_list, color_list, sha, col):
    sc_list = [i for i, s, c in zip(image_list, label_list, color_list) if sha == s and col in c]
    sc_list = np.array(sc_list)
    visualize_retrieval(sc_list, 10)



def get_shape_accuracy(knn_labels, gt):
    unique_values, counts = np.unique(knn_labels == gt, return_counts=True)
    numbers = counts[1] if len(counts) > 1 else 0
    percentage_hits = (np.float64(numbers) / np.float64(len(knn_labels))) * np.float64(100)
    return percentage_hits



def get_color_accuracy(lab_kmeans, lab_gt):
    correcte = 0
    for c, z in zip(lab_kmeans, lab_gt):
        cont = sum(1 for a, b in zip(c, z) if np.array_equal(a, b))
        correcte += cont / len(z)
    return correcte / len(lab_gt)


if __name__ == '_main_':
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # You can start coding your functions here

    kmeans = km.KMeans(test_imgs[30], 2)
    kmeans.fit()

    knn = kn.KNN(train_imgs, train_class_labels)

    shape = 'Shirts'
    Retrieval_by_shape(test_imgs, test_class_labels, shape)

    color = 'Blue'
    Retrieval_by_color(test_imgs, test_color_labels, color)

    Retrieval_combined(test_imgs, test_class_labels, test_color_labels, shape, color)
    
    knn.predict(train_imgs, 60)
    perc = get_shape_accuracy(knn.get_class(), test_class_labels)
    print(perc)