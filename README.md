# Reviews

To run the model, use the run notebook.

To run the tests, open a terminal in the root directory and use the following commands:

coverage run -m pytest tests/evaluate_test.py tests/pca_test.py tests/knn_test.py

coverage report

The tests should be passed and the report will show the code coverage of the tests.
