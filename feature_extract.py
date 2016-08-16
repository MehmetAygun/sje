import file_helper as fh
import os.path
import numpy as np
import caffe
from keras.utils.generic_utils import Progbar
import  pandas as pd
import h5py

if __name__ == '__main__':

    """ SET CAFFE PARAMETERS """

    #image_files
    SOURCE_DIR = "/storage/mehmet/Zero-Shot/datasets/CUB_200_2011/CUB_200_2011/images"
    #model_file
    MODEL_FILE = '/storage/mehmet/Models/ResNet-152-deploy.prototxt'
    #pretrained_model
    PRETRAINED = '/storage/mehmet/Models/ResNet-152-model.caffemodel'

    net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 255.0)

    # set gpu mode
    caffe.set_mode_gpu()

    pb = Progbar(11788)

    feature_data_train = {'image_name': [], 'class': [], 'pool5': []}
    feature_data_valid = {'image_name': [], 'class': [], 'pool5': []}
    feature_data_test = {'image_name': [], 'class': [], 'pool5': []}

    #train-valid-test class list directories
    train_txt = "/storage/mehmet/Zero-Shot/datasets/CUB_200_2011/CUB_200_2011/trainclasses.txt"
    valid_txt = "/storage/mehmet/Zero-Shot/datasets/CUB_200_2011/CUB_200_2011/valclasses.txt"
    test_txt = "/storage/mehmet/Zero-Shot/datasets/CUB_200_2011/CUB_200_2011/testclasses.txt"

    #read classes
    with open(train_txt) as f:
        list_train = []
        for line in f:
            list_train.append(line[:-1])

    with open(valid_txt) as f:
        list_valid = []
        for line in f:
            list_valid.append(line[:-1])

    with open(test_txt) as f:
        list_test = []
        for line in f:
            list_test.append(line[:-1])


    for imagePath in fh.list_images(SOURCE_DIR):
        image_name = imagePath.split("/")[-1]
        class_name = imagePath.split("/")[-2]

        #read image for caffe
        img = caffe.io.load_image(imagePath)
        #process image
        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        out = net.forward()

        pb.add(1)

        #check is image inside in train/valid/test 
        if class_name in list_train:
            feature_data_train['image_name'].append(image_name)
            feature_data_train['class'].append(class_name)
            feature_data_train['pool5'].append(net.blobs['pool5'].data.copy())

        elif class_name in list_valid:
            feature_data_valid['image_name'].append(image_name)
            feature_data_valid['class'].append(class_name)
            feature_data_valid['pool5'].append(net.blobs['pool5'].data.copy())

        else:
            feature_data_test['image_name'].append(image_name)
            feature_data_test['class'].append(class_name)
            feature_data_test['pool5'].append(net.blobs['pool5'].data.copy())


    # save datas
    h5file_train = h5py.File("/storage/mehmet/Zero-Shot/Features/train.data", "w")
    h5file_valid = h5py.File("/storage/mehmet/Zero-Shot/Features/valid.data", "w")
    h5file_test = h5py.File("/storage/mehmet/Zero-Shot/Features/test.data", "w")


    h5file_train.create_dataset('image_name', data=np.array(feature_data_train['image_name']))
    h5file_train.create_dataset('class', data=np.array(feature_data_train['class']))
    h5file_train.create_dataset('pool5', data=np.array(feature_data_train['pool5']))

    h5file_valid.create_dataset('image_name' ,data= np.array(feature_data_valid['image_name']))
    h5file_valid.create_dataset('class' ,data= np.array(feature_data_valid['class']))
    h5file_valid.create_dataset('pool5',data= np.array(feature_data_valid['pool5']))

    h5file_test.create_dataset('image_name',data = np.array(feature_data_test['image_name']))
    h5file_test.create_dataset('class', data = np.array(feature_data_test['class']))
    h5file_test.create_dataset('pool5', data= np.array(feature_data_test['pool5']))

    h5file_train.close()
    h5file_valid.close()
    h5file_test.close()

