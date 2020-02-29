# Reviews

To run the model, use the run notebook.

To run the tests, open a terminal in the root directory and use the following commands:

coverage run -m pytest tests/test_evaluate.py tests/test_pca.py tests/test_knn.py

coverage report

The tests should be passed and the report will show the code coverage of the tests.
