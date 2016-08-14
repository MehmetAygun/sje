import  numpy as np

def read_attributes(file_path):
    #200 class, 312 attributes
    attributes = np.zeros(shape=(200,312))

    with open(file_path) as f:
        data = f.readlines()

    i = 0
    j = 0
    for line in data:
        j = 0
        line_copy = line.split(" ")
        for att in line_copy:
            attributes[i][j] = float(att)
            j = j + 1
        i = i + 1
    return  attributes




if __name__ == '__main__':
    list = read_attributes("/storage/mehmet/Zero-Shot/datasets/CUB_200_2011/CUB_200_2011/attributes/class_attribute_labels_continuous.txt")

    print  list.shape
