#Author : Mehmet Aygun
import h5py
import numpy as np
from keras.utils.generic_utils import Progbar
from sklearn.preprocessing import normalize

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

    # Normalize projected vector
    projected_vector = normalize(projected_vector.reshape(1, -1), axis=1, norm='l2')

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

    else:
        h5file = h5py.File("/storage/mehmet/Zero-Shot/Features/test.data", "r")

    attributes = np.loadtxt("/storage/mehmet/Zero-Shot/datasets/CUB_200_2011/CUB_200_2011/attributes/class_attribute_labels_continuous.txt")

    # change class index to int value
    classes = []

    features = np.array(h5file['pool5'][:])
    number_of_examples = features.shape[0]
    feature_dimension = features.shape[2]
    features = features.reshape(number_of_examples, feature_dimension)  # 2048 feature dimension

    for i in range(0, number_of_examples):
        classes.append(int(h5file['class'][i][0]) * 100 + int(h5file['class'][i][1]) * 10 + int(h5file['class'][i][2]))

    classes = np.array(classes)

    pb = Progbar(number_of_examples)
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