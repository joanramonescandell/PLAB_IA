_authors_ = 'TO_BE_FILLED'
_group_ = 'TO_BE_FILLED'

import numpy as np
from utils_data import read_dataset, visualize_retrieval
import Kmeans as km
import KNN as kn


def retrieval_by_color(image_list, label_list, col):
    '''Function that receives as input a list of images, the tags we obtained
    by applying the Kmeans algorithm to those images, and the question we ask for a
    specific search (that is, a string or list of strings with the colors we want to look
    for). It returns all images that contain the question tags we ask. This feature can be
    improved by adding an input parameter that contains the percentage of each color in
    the image, and returns the ordered images'''
    # We create a list of lists with the images that contain the colors we ask for
    images = []
    for i in range(len(image_list)):
        for j in range(len(col)):
            if col[j] in label_list[i]:
                images.append(image_list[i])
                break
    return images  


def retrieval_by_shape(image_list, label_list, sha):
    ''' Function that receives as input a list of images, the tags we obtained
    by applying the KNN algorithm to these images and the question we ask for a specific
    search (that is, a string defining the shape of clothes we want to search). It returns all
    images that contain the question tag we ask. This feature can be enhanced by adding
    an input parameter that contains the percentage of K-neighbors with the tag we are
    looking for and returns the ordered images.'''
    # We create a list of lists with the images that contain the shape we ask for
    images = []
    for i in range(len(image_list)):
        if sha in label_list[i]:
            images.append(image_list[i])
    return images



def retrieval_combined(image_list, label_list, color_list, sha, col):
    '''Function that receives as input a list of images, shape and color
    tags, a shape question, and a color question. It returns images that match the two
    questions, for example: Red Flip Flops. As in the previous functions, this function can
    be improved by entering the color and shape percentage data of the labels.'''
    # We create a list of lists with the images that contain the shape and color we ask for
    images = []
    for i in range(len(image_list)):
        if sha in label_list[i] and col in label_list[i]:
            images.append(image_list[i])
    return images

def get_shape_accuracy(knn_labels, gt):
    '''Function that receives as input the tags we obtained when applying
    the KNN and the Ground-Truth of these. Returns the correct tag percentage'''
    # We create a list of lists with the images that contain the shape and color we ask for
    correct = 0
    for i in range(len(knn_labels)):
        if knn_labels[i] == gt[i]:
            correct += 1
    return correct / len(knn_labels) 

def get_color_accuracy(lab_kmeans, lab_gt):
    '''Function that receives as input the tags we obtained when applying
    the kmeans and the Ground-Truth of these. Returns the correct tag percentage. Keep
    in mind that we can have more than one tag for each image, so you need to think about
    how to score if the prediction and the Ground-Truth partially match. In the theory
    class you were given some ideas for measuring the similarity between these sets.'''
    # We create a list of lists with the images that contain the shape and color we ask for
    correct = 0
    for i in range(len(lab_kmeans)):
        for j in range(len(lab_kmeans[i])):
            if lab_kmeans[i][j] == lab_gt[i][j]:
                correct += 1
    return correct / len(lab_kmeans)


if __name__ == '__main__':
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
    test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    # Apply Kmeans to the train images
    kmeans = km.KMeans(train_imgs, K=10)
    kmeans.fit()
    # Get the labels for the test images
    test_labels = kmeans.predict(test_imgs)
    shape = 'Shirt'
    color = 'Red'
    # Get the images that match the shape and color
    images = retrieval_combined(test_imgs, test_labels, test_color_labels, shape, color)
    # Visualize the results
    visualize_retrieval(images, topN=5, info=None, ok=None, title='Retrieval by shape and color', query=None, fig_name="fig.png")
    # Get the accuracy of the shape and color
    shape_accuracy = get_shape_accuracy(test_labels, test_class_labels)
    color_accuracy = get_color_accuracy(test_color_labels, test_color_labels)
    print('Shape accuracy: ', shape_accuracy)
    print('Color accuracy: ', color_accuracy)
    