#Author : Mehmet Aygun

"""
 SJE Training
 It learns a compability function between  CNN features and attribute output label embeddings,
 for fine-grained zero-shot image classification.
 Used Dataset : Caltech-UCSD Birds-200-2011 Dataset

"""
from sklearn.preprocessing import normalize
import numpy as np
import read_attributes as ra
import h5py
import test
import math
from random import shuffle
import  random
random.seed( 10 )
from keras.utils.generic_utils import Progbar

RAND_MAX = 2147483647

def rand_gen():
    """
    It create a random number and return that number.
    """
    a = 0.0
    b = 0.0

    while a == 0:
        a = float(float(random.random())/float(RAND_MAX))
    while b == 0:
        b = float(float(random.random())/float(RAND_MAX))
    return (math.sqrt(-2.0 * math.log(a)) * math.cos(2*math.pi*b))

def argmax(input_embedding,W,output_embeddings,correct_index):
    """
            This function takes a input embedding and returns class index which have higher compatibility score
            input = nX1
            output_embeddings = number_of_classes X attribute_dimension
            W = n X attribute_dimension

    """
    return_index = -1
    max_score = 0.0

    # Project image features on embedding matrix
    #projected_vector = np.dot(input_embedding,W)
    projected_vector = np.array(input_embedding).dot(np.matrix(W))

    # Normalize projected vector
    projected_vector = normalize(projected_vector.reshape(1,-1), axis=1,norm='l2')


    #Compare projected vector for finding high compatability score
    for i in range(0,output_embeddings.shape[0]):
        # dot product similarity
        score = np.dot(projected_vector, output_embeddings[i])
        #if image correct label is false add a margin
        if i != correct_index:
            cost = 5.0
        else:
            cost = 0.0

        score = score + cost #margin
        if score > max_score:
            max_score = score
            return_index = i

    #return index that create maximum compatibility score
    return return_index

if __name__ == '__main__':

    #read attributes
    attributes = ra.read_attributes("/storage/mehmet/Zero-Shot/datasets/CUB_200_2011/CUB_200_2011/attributes/class_attribute_labels_continuous.txt")

    number_of_classes = attributes.shape[0]
    attributes_dimension = attributes.shape[1]

    #read features
    h5file_train = h5py.File("/storage/mehmet/Zero-Shot/Features/train.data", "r")

    # 5490 item
    train_image_name = np.array(h5file_train['image_name'][:])

    number_of_images_train = train_image_name.size
    #change class index to int value
    train_class = []

    #create random index list for training
    random_index = [[i] for i in range(number_of_images_train)]
    shuffle(random_index)

    #change class indexes string to integer
    for i in range(0,number_of_images_train):
        train_class.append(int(h5file_train['class'][i][0])*100 + int(h5file_train['class'][i][1])*10 +int(h5file_train['class'][i][2]))

    #read image features
    train_features = np.array(h5file_train['pool5'][:])
    train_features_dimension = train_features.shape[2]
    train_features = train_features.reshape(number_of_images_train,train_features_dimension)


    train_class = np.array(train_class)

    #Parameters that should be validate
    learning_rate = 1e-7
    max_epoch = 20
    lda = 1e-63


    #Randomly initialize embedding Matrix W

    std_dev = 1.0 / math.sqrt(attributes_dimension)
    W =np.zeros((train_features_dimension,attributes_dimension))
    for i in range(train_features_dimension):
        dot = 0
        for j in range(attributes_dimension):
            W[i][j] = std_dev * rand_gen()
        W[i] = normalize(W[i].reshape(1,-1),axis=1,norm='l2')

    # W_best hold the weights that have best accuracy on validation set
    W_best = np.copy(W)
    best_accuracy = 0

    #Start training
    for i in range(0,max_epoch):
        number_of_true = 0
        print "Epoch " + str(i)
        pb = Progbar(number_of_images_train)

        for j in random_index:
            pb.add(1)
            y = argmax(train_features[j],W,attributes,train_class[j]-1)
            if (train_class[j]-1) != y and y!=-1: # if wrong predicton make update
                W = W + np.dot(np.transpose(train_features[j]), (attributes[train_class[j]-1]-attributes[y])) * learning_rate
            elif (train_class[j]-1) == y:
                number_of_true += 1
        print "Number of true " + str(number_of_true)

        #Get validation accuracy
        print "Testing : "
        accuracy = test.get_accuracy(W,valid=True)
        print "Epoch " + str(i) + "accuracy is :" + str(accuracy)
        if accuracy > best_accuracy :
            best_accuracy = accuracy
            W_best = np.copy(W)
        print "Best accuracy so far is :" + str(best_accuracy)

    #Save Model
    print "Optimization Done! \n Saving Model..."
    h5file_train = h5py.File("/storage/mehmet/Zero-Shot/Models/sje_w.data", "w")
    h5file_train.create_dataset('W', data=W_best)
    h5file_train.close()

    print "All Done !"





