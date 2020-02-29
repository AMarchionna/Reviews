from reviews.evaluate import Evaluate


def test_eval():
    '''
    In this test we test the evaluate module. The dataset
    contains some positive reviews that have words such as 'good'
    and 'nice', and some negative reviews that have words such as
    'bad'. The testing reviews also have just this words, so the 
    vectors should be very separated and the score should be 1.
    '''
    # The number of reviews to train and test the model.
    MAX_TRAIN = 5
    MAX_TEST = 3

    components = 5  # The number of components for the PCA model.
    # The number of neighbours for the KNN model. As different vectors are
    # very far from each other, the closest one should be of the same
    # category. 
    neighbours = 1

    path = "data/mini_test.csv"  # Path to the dataset.

    model = Evaluate(MAX_TRAIN, MAX_TEST, max_df=1.0, min_df=0.4)  # Set the model.
    model.read(path)  # Read the data.
    model.set_PCA(components)  # Set the PCA class.
    model.set_KNN(neighbours)  # Set the KNN class.
    score = model.score()  # Obtain the score of the model.

    assert score == 1.0
