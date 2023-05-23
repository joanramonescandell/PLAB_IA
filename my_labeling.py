import numpy as np
from utils_data import read_dataset, visualize_retrieval
import Kmeans as km
import KNN as kn


def Retrieval_by_color(image_list, label_list, col):
    col_list = [i for i, et in zip(image_list, label_list) if col in et]
    visualize_retrieval(col_list, 10)


def Retrieval_by_shape(image_list, label_list, sha):
    sha_list = [i for i, et in zip(image_list, label_list) if et == sha]
    visualize_retrieval(sha_list, 10)


def Retrieval_combined(image_list, label_list, color_list, sha, col):
    sc_list = [i for i, s, c in zip(image_list, label_list, color_list) if sha == s and col in c]
    visualize_retrieval(sc_list, 10)


def get_shape_accuracy(knn_labels, gt):
    numbers = np.unique(knn_labels == gt, return_counts=True)[1][1]
    percentage_hits = (np.float64(numbers) / np.float64(len(knn_labels))) * np.float64(100)
    return percentage_hits



def get_color_accuracy(kmeans_labels, gt):
    hits = 0
    num_labels = len(kmeans_labels)

    for pos in range(num_labels):
        unique_colors = set(kmeans_labels[pos])
        count = sum(1 for col, ct in zip(kmeans_labels[pos], gt[pos]) if col == ct)
        hits += count / len(gt[pos])

    percentage_hits = hits / num_labels * 100
    
    return percentage_hits





if __name__ == '__main__':
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset()

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    kmeans = km.KMeans(test_imgs[30], 2)
    kmeans.fit()

    knn = kn.KNN(train_imgs, train_class_labels)

    shape = 'Shirts'
    Retrieval_by_shape(test_imgs, test_class_labels, shape)

    color = 'Blue'
    Retrieval_by_color(test_imgs, test_color_labels, color)

    Retrieval_combined(test_imgs, test_class_labels, test_color_labels, shape, color)

    # Shape accuracy test
    shape_acc = kn.KNN(train_imgs, train_class_labels)
    shape_acc.predict(test_imgs, 30)
    shape_percent = get_shape_accuracy(shape_acc.get_class(), test_class_labels)
    print("Shape accuracy : ", round(shape_percent, 2), "%")
