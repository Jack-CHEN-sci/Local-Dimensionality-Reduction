# import the dataset
from sklearn.datasets import load_breast_cancer
'''
    Breast Cancer Data-set Introduction
    The Breast Cancer data set is a real-valued multivariate data that consists of two classes, 
    where each class signifies whether a patient has breast cancer or not. 
    The two categories are: malignant and benign.

    The malignant class has 212 samples, whereas the benign class has 357 samples.
    It has 30 features shared across all classes: radius, texture, perimeter, area, smoothness, fractal dimension, etc.
'''
import numpy as np
import pandas as pd


def main():
    # Load data-set
    breast = load_breast_cancer()
    # Fetch the data
    breast_data = breast.data
    # print("Shape of the data:", breast_data.shape) == (569, 30)
    # Fetch the labels
    breast_labels = breast.target
    # print("Shape of the data labels:", breast_labels.shape) == (569, )
    '''
        Difference between shape (x, ) and (x,1):
        (x, ): 1D array, with 2 elements in
        (x,1): xD array, with 1 element each line
    '''

    # Reshape the breast_labels to concatenate it with the breast_data
    labels = np.reshape(breast_labels, (569, 1))
    # Concatenate the data and labels along the second axis
    final_breast_data = np.concatenate([breast_data, labels], axis=1)
    # print(final_breast_data)

    # Create the DataFrame of the final data to represent the data in a tabular fashion
    breast_dataset = pd.DataFrame(final_breast_data)

    features = breast.feature_names
    # print(features)
    # Manually add "label" field to the features array
    features_labels = np.append(features, 'label')

    # Embed the column names to the breast data-set data-frame
    breast_dataset.columns = features_labels
    print(breast_dataset.head())

    # Change the labels from "0 and 1" to "benign and malignant"
    breast_dataset['label'].replace(0, 'Benign', inplace=True)
    breast_dataset['label'].replace(1, 'Malignant', inplace=True)
    print(breast_dataset.tail())


if __name__ == '__main__':
    main()
