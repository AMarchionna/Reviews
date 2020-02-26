from knn import KNNClassifier
from pca import PCA
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class Evaluate:
    def __init__(self, max_train=6225, max_test=6275):
        '''Function that trains and tests the model. '''
        self.max_train = min(max_train, 6225)
        self.max_test = min(max_test, 6275)

    def read(self):
        '''Function that reads the files and sets everything
           up so that the model can be trained.'''
        print("Reading files...")
        # Read the file with the reviews.
        df = pd.read_csv("../data/imdb_small.csv", index_col=0)
        print("Number of reviews: {}".format(df.shape[0]))

        # Set the dataframe with the training reviews
        text_train = df[df.type == 'train']["review"]
        label_train = df[df.type == 'train']["label"]
        # Cut some reviews if needed
        text_train = text_train[0: self.max_train]
        label_train = label_train[0: self.max_train]
        # Set the dataframe for the testing reviews
        text_test = df[df.type == 'test']["review"]
        label_test = df[df.type == 'test']["label"]
        # Cut some reviews if needed
        text_test = text_test[0: self.max_test]
        label_test = label_test[0: self.max_test]

        print("Number of training reviews = {}".
              format(len(text_train)))
        print("Number of test reviews = {}".format(len(text_test)))

        # Pick the word phrases that appear in reviews.
        vectorizer = CountVectorizer(max_df=0.90, min_df=0.01,
                                     max_features=500, ngram_range=(1, 2))
        vectorizer.fit(text_train)
        # Change reviews into vectors
        X_train = vectorizer.transform(text_train)
        self.y_train = (label_train == 'pos').values
        X_test, self.y_test = vectorizer.transform(text_test), (label_test ==
                                                                'pos').values

        self.X_train = X_train.toarray()
        self.X_test = X_test.toarray()

    def set_PCA(self, components):
        '''Performs PCA to the datasets.'''
        print("Setting PCA...")
        self.components = components
        # Set the PCA class.
        pca = PCA(components)
        pca.fit(self.X_train)
        # Transform with the PCA class the two datasets.
        self.X_train_red = pca.transform(self.X_train)
        self.X_test_red = pca.transform(self.X_test)

    def set_KNN(self, neighbours):
        '''Sets the KNN class.'''
        print("Setting KNN...")
        self.neighbours = neighbours
        # Set the KNN class.
        self.knn = KNNClassifier(neighbours)
        self.knn.fit(self.X_train_red, self.y_train)

    def score(self):
        '''Calculates the accuracy of the model.'''
        print("Calculating the score of the model...")
        score = self.knn.score(self.X_test_red, self.y_test)
        print("The accuracy when the number of components is {:d} and the"
              .format(self.components),
              "number of neighbours is {:d}, is {:.4f}"
              .format(self.neighbours, score))
        return score
