'''

Authors: Nishant Kumar.

Copyright:
Copyright (c) 2020 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

**
Original code from https://github.com/deep-diver/CIFAR10-img-classification-tensorflow
Modified for our purposes.
**

'''

import os, sys
import pickle
import random
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

preProcessedImgSaveFolderConst = './PreProcessedImages'

def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

def display_stats(cifar10_dataset_folder_path, batch_id, sample_id, savepng=False, showfig=False):
    features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)

    if not (0 <= sample_id < len(features)):
        print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))
        return None

    print('\nStats of batch #{}:'.format(batch_id))
    print('# of Samples: {}\n'.format(len(features)))

    label_names = load_label_names()
    label_counts = dict(zip(*np.unique(labels, return_counts=True)))
    for key, value in label_counts.items():
        print('Label Counts of [{}]({}) : {}'.format(key, label_names[key].upper(), value))

    sample_image = features[sample_id]
    sample_label = labels[sample_id]

    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))

    if savepng or showfig:
        # Save/show a .png file for the current image
        plt.imshow(sample_image)
        if savepng:
            plt.savefig('foo.png')
        elif showfig:
            plt.show()

def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def one_hot_encode(x):
    """
        argument
            - x: a list of labels
        return
            - one hot encoding matrix (number of labels, number of class)
    """
    encoded = np.zeros((len(x), 10))

    for idx, val in enumerate(x):
        encoded[idx][val] = 1

    return encoded

def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    features = normalize(features)
    labels = one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))

# Saved files are 'preprocess_batch_' + str(batch_i) + '.p',
#                 'preprocess_validation.p',
#                 'preprocess_testing.p'
def preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode, preProcessedImgSaveFolder = preProcessedImgSaveFolderConst):
    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)

        # find index to be the point as validation data in the whole dataset of the batch (10%)
        index_of_validation = int(len(features) * 0.1)

        # preprocess the 90% of the whole dataset of the batch
        # - normalize the features
        # - one_hot_encode the lables
        # - save in a new file named, "preprocess_batch_" + batch_number
        # - each file for each batch
        _preprocess_and_save(normalize, one_hot_encode,
                             features[:-index_of_validation], labels[:-index_of_validation],
                             os.path.join(preProcessedImgSaveFolder, 'preprocess_batch_' + str(batch_i) + '.p'))

        # unlike the training dataset, validation dataset will be added through all batch dataset
        # - take 10% of the whold dataset of the batch
        # - add them into a list of
        #   - valid_features
        #   - valid_labels
        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])

    # preprocess the all stacked validation dataset
    _preprocess_and_save(normalize, one_hot_encode,
                         np.array(valid_features), np.array(valid_labels),
                         os.path.join(preProcessedImgSaveFolder, 'preprocess_validation.p'))

    # load the test dataset
    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # preprocess the testing data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Preprocess and Save all testing data
    _preprocess_and_save(normalize, one_hot_encode,
                         np.array(test_features), np.array(test_labels),
                         os.path.join(preProcessedImgSaveFolder, 'preprocess_testing.p'))

def get_one_sample_point(batch_id, sample_id, preProcessedImgSaveFolder = preProcessedImgSaveFolderConst):
    features, labels = pickle.load(open(os.path.join(preProcessedImgSaveFolder, 'preprocess_batch_' + str(batch_id) + '.p'), mode='rb'))
    return (features[sample_id], labels[sample_id])

def get_sample_points(batch_id, start_id, end_id, preProcessedImgSaveFolder = preProcessedImgSaveFolderConst):
    features, labels = pickle.load(open(os.path.join(preProcessedImgSaveFolder, 'preprocess_batch_' + str(batch_id) + '.p'), mode='rb'))
    return (features[start_id:end_id], labels[start_id:end_id])

def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

def load_preprocess_training_batch(batch_id, batch_size, preProcessedImgSaveFolder = preProcessedImgSaveFolderConst):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = os.path.join(preProcessedImgSaveFolder, 'preprocess_batch_' + str(batch_id) + '.p')
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)

def load_preprocess_training_data(batch_id, preProcessedImgSaveFolder = preProcessedImgSaveFolderConst):
    filename = os.path.join(preProcessedImgSaveFolder, 'preprocess_batch_' + str(batch_id) + '.p')
    features, labels = pickle.load(open(filename, mode='rb'))
    return features, labels

def load_preprocess_validation_data(preProcessedImgSaveFolder = preProcessedImgSaveFolderConst):
    valid_features, valid_labels = pickle.load(open(os.path.join(preProcessedImgSaveFolder, 'preprocess_validation.p'), mode='rb'))
    return valid_features, valid_labels

def load_preprocess_testing_data(preProcessedImgSaveFolder = preProcessedImgSaveFolderConst):
    testing_features, testing_labels = pickle.load(open(os.path.join(preProcessedImgSaveFolder, 'preprocess_testing.p'), mode='rb'))
    return testing_features, testing_labels

def main():
    cifar10_dataset_folder_path = '../../HelperScripts/CIFAR10/cifar-10-batches-py'
    preProcessedImgSaveFolder = './PreProcessedImages'
    preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)
    display_stats(cifar10_dataset_folder_path, 2, 4555)
    print(get_one_sample_point(2, 4555))

if __name__ == '__main__':
    main()
