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
import test
import math
from random import shuffle
import  random
random.seed( 10 )
from keras.utils.generic_utils import Progbar

RAND_MAX = 2147483647

def rand_gen():
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

    W = np.matrix(W)
    projected_vector = np.dot(input_embedding,W)
    #normalize
    projected_vector /= math.sqrt(np.dot(projected_vector,projected_vector.transpose()))

    for i in range(0,output_embeddings.shape[0]):
        score = np.dot(projected_vector, output_embeddings[i]) # dot product similarity
        if i != correct_index:
            cost = 1.0
        else:
            cost = 0.0
        score += cost #margin
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

    random_index = [[i] for i in range(5490)]

    for i in range(0,5490):
        train_class.append(int(h5file_train['class'][i][0])*100 + int(h5file_train['class'][i][1])*10 +int(h5file_train['class'][i][2]))
    train_features = np.array(h5file_train['pool5'][:]).reshape(5490,2048)#2048 feature dimension
    train_class = np.array(train_class)



    #Parameters that should be validate
    learning_rate = 1e-5
    max_epoch = 20
    lda = 1e-63
    #initialize W

    std_dev = 1.0 / math.sqrt(312)

    W =np.zeros((2048,312))# np.random.random_sample(2048*312)# 2048:feature dim,312:a output embedding dimension
    for i in range(2048):
        dot = 0
        for j in range(312):
            W[i][j] = std_dev * rand_gen()
            dot += (W[i][j])*(W[i][j])
        W[i] *= (lda/dot)

    W_best = np.copy(W)
    best_accuracy = 0

    for i in range(0,max_epoch):
        number_of_true = 0
        print "Epoch " + str(i)
        pb = Progbar(500)
        shuffle(random_index)
        for j in random_index[0:500]:

            pb.add(1)
            y = argmax(train_features[j],W,attributes,train_class[j]-1)
            if (train_class[j]-1) != y and y!=-1: #make update
                W = W + np.dot(np.transpose(train_features[j]),learning_rate * (attributes[train_class[j]-1]-attributes[y]))
            else:
                number_of_true += 1
        print "Number of true " + str(number_of_true)

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





