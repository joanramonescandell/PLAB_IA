import numpy as np
import Kmeans as km
from utils_data import read_dataset, visualize_k_means, visualize_retrieval, Plot3DCloud



def retrieval_by_color(image_list, label_list, col):
    # recibe lista imágenes,lista etiquetas y un color.
    # Genera nueva lista con  imágenes que contienen el color .
    col_list = [i for i, et in zip(image_list, label_list) if col in et]

    visualize_retrieval(col_list, 10)


def retrieval_by_shape(image_list, label_list, sha):
    # recibe lista imágenes,lista etiquetas y una forma.
    # Genera nueva lista con  imágenes que contienen la forma .
    sha_list = [i for i, et in zip(image_list, label_list) if et == sha]

    visualize_retrieval(sha_list, 10)


def retrieval_combined(image_list, label_list, color_list, sha, col):
    #  recibe lista imágenes,lista etiquetas, lista de colores una forma y un color.

    # Genera una nueva lista con las imágenes que tienen la forma y color especificados.
    sc_list = [i for i, s, c in zip(image_list, label_list, color_list) if sha == s and col in c]

    visualize_retrieval(sc_list, 10)


def get_shape_accuracy(knn_labels, gt):
    # recibe las etiquetas predichas por el algoritmo k-NN y las etiquetas verdaderas (gt).
    # Cuenta cuántas de las etiquetas k-NN coinciden con las etiquetas verdaderas.
    numbers = np.unique(knn_labels == gt, return_counts=True)[1][1]

    # porcentaje de aciertos dividiendo el número de coincidencias entre el total de etiquetas.
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
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels = read_dataset()

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Initialize KMeans with the test images
    kmeans = km.KMeans(test_imgs[30], 2)
    kmeans.fit()
    '''
    # Initialize KNN with the train images and their class labels
    knn = kn.KNN(train_imgs, train_class_labels)
    # Choose the shape to retrieve
    shape = 'Dresses'
    retrieval_by_shape(test_imgs, test_class_labels, shape)

    # Choose the color to retrieve
    color = 'Blue'
    retrieval_by_color(test_imgs, test_color_labels, color)

    # Combine shape and color retrieval
    retrieval_combined(test_imgs, test_class_labels, test_color_labels, shape, color)

    # Shape accuracy test
    shape_acc = kn.KNN(train_imgs, train_class_labels)
    shape_acc.predict(test_imgs, 60)
    shape_percent = get_shape_accuracy(shape_acc.get_class(), test_class_labels)
    print("Shape accuracy : ", round(shape_percent, 2), "%")
    '''

    visualize_k_means(kmeans, [80, 60, 3])
    Plot3DCloud(kmeans)
