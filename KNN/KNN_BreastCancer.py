import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns


def main():
    sns.set()
    # load brest cancer data-set
    breast_cancer = load_breast_cancer()
    # Create the DataFrame of the final data to represent the data in a tabular fashion
    X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    # Use only two features
    X = X[['mean area', 'mean compactness']]

    # Make a Categorical type from codes and categories or dtype.
    y = pd.Categorical.from_codes(breast_cancer.target, breast_cancer.target_names)
    # One hot
    y = pd.get_dummies(y, drop_first=True)
    print(y)

    # Randomly divide data-set into training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Draw a plot
    plt.figure(figsize=[10, 5])

    plt.subplot(121)
    sns.scatterplot(
        x='mean area',
        y='mean compactness',
        hue='benign',
        data=X_test.join(y_test, how='outer')
    )

    plt.subplot(122)
    plt.scatter(
        X_test['mean area'],
        X_test['mean compactness'],
        c=y_pred,
        cmap='coolwarm',
        alpha=0.7
    )

    plt.show()


if __name__ == '__main__':
    main()
