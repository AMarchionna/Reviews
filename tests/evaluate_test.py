import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, '/home/agustin/git_repo/Reviews')
from reviews.evaluate import Evaluate


def test_eval():
    MAX_TRAIN = 5
    MAX_TEST = 3

    components = 5
    neighbours = 1

    path = "data/mini_test.csv"

    model = Evaluate(MAX_TRAIN, MAX_TEST, max_df=1.0, min_df=0.4)
    model.read(path)
    model.set_PCA(components)
    model.set_KNN(neighbours)
    score = model.score()

    assert score == 1.0
