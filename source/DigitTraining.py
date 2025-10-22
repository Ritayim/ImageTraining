from collections import Counter
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os

def most_common_element(numpy_array):
    uniques, counts = np.unique(numpy_array, return_counts=True)
    max_index = np.argmax(counts)
    return np.array([uniques[max_index]])

### images_test: matrix of shape (1000, 784)
### labels_test: matrix of shape (1000, 1)
### images_train: matrix of shape (random, 784)
### labels_train: matrix of shape (random, 1)
def kNN(images_train, labels_train, images_test, labels_test, k):
    acc = np.zeros(10)
    class_acc = np.zeros(10)
    class_total = np.zeros(10)

    for i in range(len(images_test)): #n
        image_test = images_test[i]
        label_test = labels_test[i]

        distances = np.linalg.norm(images_train - image_test, axis=1)
        indices = np.argsort(distances)[:k]
        nearest_neighbors = labels_train[indices]
        predicted_label = most_common_element(nearest_neighbors)

        class_total[int(label_test[0])] += 1
        if np.array_equal(predicted_label, label_test):
            class_acc[int(label_test[0])] += 1
        
        # print(f'Predicted: {predicted_label[0]}, Actual: {label_test[0]}')
    
    for c in range(10):
        if class_total[c] > 0:
            acc[c] = class_acc[c] / class_total[c]

    return acc, np.mean(acc)

def plot_for_one_k(images_train, labels_train, images_test, labels_test):
    x, y = [], []
    rand_val = np.random.randint(30, 10000, size=10)
    for val in rand_val:
        images_train_rand = images_train[0:val, :]
        labels_train_rand = labels_train[0:val, :]
        acc, acc_av = kNN(images_train_rand, labels_train_rand, images_test, labels_test, k=1)
        x.append(val)
        y.append(acc_av)

    plt.figure(1)
    plt.title('Accuracy vs Number of Training Samples (k=1)')
    plt.xlabel('Number of training data')
    plt.ylabel('Accuracy')
    plt.xlim(0, 10000)
    plt.plot(x, y, marker='o')

def plot_for_diff_k(images_train, labels_train, images_test, labels_test):
    rand_val = np.random.randint(30, 10000, size=10)
    k_values = [1, 3, 5, 10]
    plt.figure(2)

    for k in k_values:
        x, y = [], []
        for val in rand_val:
            images_train_rand = images_train[0:val, :]
            labels_train_rand = labels_train[0:val, :]
            acc, acc_av = kNN(images_train_rand, labels_train_rand, images_test, labels_test, k)
            x.append(val)
            y.append(acc_av)
        plt.plot(x, y, label=f'k={k}', marker='o')

    plt.title('Average accuracy vs k values')
    plt.legend()
    plt.xlabel('Number of training data')
    plt.ylabel('Average accuracy')

def plot_best_k(images_data, labels_data):
    k_values = [1, 3, 5, 10]
    x, y = [], []
    images_data, labels_data = images_data[0:2000,:], labels_data[0:2000,:]
    images_train, labels_train = images_data[0:1000,:], labels_data[0:1000,:]
    images_test, labels_test = images_data[1000:2000,:], labels_data[1000:2000,:]

    for k in k_values:
        acc, acc_av = kNN(images_train, labels_train, images_test, labels_test, k)
        x.append(k)
        y.append(acc_av)

    plt.figure(3)
    plt.title('Average accuracy vs k values (on 2000 samples)')
    plt.xlabel('k values')
    plt.ylabel('Average accuracy')
    plt.xlim(0, 11)
    plt.plot(x, y, marker='o')

def main():
    #Loading the data
    directory = os.path.dirname(os.path.abspath(__file__)) + '/'
    M = loadmat(directory + 'MNIST_digit_data.mat')
    images_train,images_test,labels_train,labels_test= M['images_train'],M['images_test'],M['labels_train'],M['labels_test']

    #just to make all random sequences on all computers the same.
    np.random.seed(4)

    #randomly permute data points
    inds = np.random.permutation(images_train.shape[0])
    images_train = images_train[inds]
    labels_train = labels_train[inds]

    inds = np.random.permutation(images_test.shape[0])
    images_test = images_test[inds]
    labels_test = labels_test[inds]

    # #if you want to use only the first 1000 data points.
    # images_train = images_train[0:1000,:]
    # labels_train = labels_train[0:1000,:]
    images_test = images_test[0:1000,:]
    labels_test = labels_test[0:1000,:]

    plot_for_one_k(images_train, labels_train, images_test, labels_test)
    plot_for_diff_k(images_train, labels_train, images_test, labels_test)
    plot_best_k(images_train, labels_train)
    plt.show()

if __name__ == "__main__":
    main()