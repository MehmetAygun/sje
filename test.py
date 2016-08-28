#Author : Mehmet Aygun
import h5py
import numpy as np
import read_attributes as ra
import  math
from keras.utils.generic_utils import Progbar


def argmax(input_embedding, W, output_embeddings):
    """
            This function takes a input embedding and returns class index which have higher compatibility score
            input = nX1
            output_embeddings = number_of_classes X attribute_dimension
            W = n X attribute_dimension

        """
    return_index = 0
    max_score = -100000

    W = np.matrix(W)
    projected_vector = np.dot(input_embedding, W)
    # normalize
    projected_vector /= math.sqrt(np.dot(projected_vector, projected_vector.transpose()))

    for i in range(0, output_embeddings.shape[0]):
        score = np.dot(projected_vector, output_embeddings[i])  # dot product similarity
        if score > max_score:
            max_score = score
            return_index = i

    return return_index


def get_accuracy (W,valid = True):
    """

    :param W:  Weight matrix
    :param valid: for valid or for test
    :return: accuracy
    """

    if valid:
        h5file = h5py.File("/storage/mehmet/Zero-Shot/Features/valid.data", "r")
        number_of_examples = 2607
    else:
        h5file = h5py.File("/storage/mehmet/Zero-Shot/Features/test.data", "r")
        number_of_examples = 3691

    attributes = ra.read_attributes("/storage/mehmet/Zero-Shot/datasets/CUB_200_2011/CUB_200_2011/attributes/class_attribute_labels_continuous.txt")
    image_names = np.array(h5file['image_name'][:])

    pb = Progbar(number_of_examples)

    # change class index to int value
    classes = []

    for i in range(0, number_of_examples):
        classes.append(int(h5file['class'][i][0]) * 100 + int(h5file['class'][i][1]) * 10 + int(h5file['class'][i][2]))

    features = np.array(h5file['pool5'][:]).reshape(number_of_examples, 2048)  # 2048 feature dimension
    classes = np.array(classes)

    correct = 0
    for j in range(0,number_of_examples):
        y = argmax(features[j],W,attributes)
        pb.add(1)
        if y == (classes[j]-1):
            correct = correct + 1

    return (correct / float(number_of_examples)) * 100


if __name__ == '__main__':
    W = h5py.File("/storage/mehmet/Zero-Shot/Models/sje_w.data", "r")['W'][:]

    print get_accuracy(W,valid=True)
