#Author : Mehmet Aygun

"""
 SJE Training
 It learns a compability function between  CNN features and attribute output label embeddings,
 for fine-grained zero-shot image classification.
 Used Dataset : Caltech-UCSD Birds-200-2011 Dataset

"""
import numpy as np
import read_attributes as ra
import h5py
from random import shuffle
from keras.utils.generic_utils import Progbar


def argmax(input_embedding,W,output_embeddings):
    """
            This function takes a input embedding and returns class index which have higher compatibility score
            input = nX1
            output_embeddings = number_of_classes X attribute_dimension
            W = n X attribute_dimension

        """
    return_index = 0
    max_score = -100000

    W= np.matrix(W)
    for i in range(0,output_embeddings.shape[0]):
        score = np.dot(np.dot(input_embedding , W) , output_embeddings[i]) # 1X1 score
        if score > max_score:
            max_score = score
            return_index = i

    return return_index

if __name__ == '__main__':

    #read attributes 200x312
    attributes = ra.read_attributes("/storage/mehmet/Zero-Shot/datasets/CUB_200_2011/CUB_200_2011/attributes/class_attribute_labels_continuous.txt")

    #read features
    h5file_train = h5py.File("/storage/mehmet/Zero-Shot/Features/train.data", "r")

    # 5490 item
    train_image_name = np.array(h5file_train['image_name'][:])

    #change class index to int value
    train_class = []

    for i in range(0,5490):
        train_class.append(int(h5file_train['class'][i][0])*100 + int(h5file_train['class'][i][1])*10 +int(h5file_train['class'][i][2]))
    train_features = np.array(h5file_train['pool5'][:]).reshape(5490,2048)#2048 feature dimension
    train_class = np.array(train_class)

    random_index = [[i] for i in range(5490)]
    shuffle(random_index)

    #Parameters that should be validate
    learning_rate = 0.001
    max_epoch = 20

    #initialize W

    W = np.random.random_sample(2048*312)# 2048:feature dim,312:a output embedding dimension
    W = W.reshape(2048,312)
    for i in range(0,max_epoch):
        print "Epoch" + str(i)
        pb = Progbar(5490)
        for j in random_index:
            pb.add(1)
            y = argmax(train_features[j],W,attributes)
            if train_class[j]-1 != y : #make update
                W = W + learning_rate* np.dot(np.transpose(train_features[j]),attributes[train_class[j]-1]-attributes[y])

    #Save Model
    print "Optimization Done! \n Saving Model..."
    h5file_train = h5py.File("/storage/mehmet/Zero-Shot/Models/sje_w.data", "w")
    h5file_train.create_dataset('W', data=W)
    h5file_train.close()

    print "All Done !"





