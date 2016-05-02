## Scikit-learn classifier speed tests

The [Scikit-learn](http://scikit-learn.org/stable/index.html) Python library provides an API for an easy-to-use set of packages for Machine Learning, including classification, regression, clustering and dimensionality reduction.  In comparing different methods, one needs to consider many factors:

+ The problem you're trying to solve.
+ The appropriateness of the method for your data.
+ The speed of the algorithm.

#### Classification
Classification is generally suited to problems where you have a number of independent variables *X*, and two or more classes in a target variable *y*.  The general problem is to predict *y* given *X*.  

Data is usually split into a training data set to fit the model, and a test data set to test how well the model and algorithm performs on unknown data.  If the test data contains the target *y* as well as independent *X's*, one may evaluate the model accuracy as to whether it can predict the correct values.

The Scikit-learn classifier API has a number of common methods for a classifier instance *clf*, including:
+ *clf.fit()* : apply to training data, fits model
+ *clf.predict()* : apply to test data, predicts *y* given *X*

#### Methods
Speed tests were performed on the *fit()* and *predict()* methods for several classifiers.  The [Human Activity Recognition Using Smartphones Data Set](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) from UC Irvine was chosen since it has 561 columns and 7352 rows.  The data was pared down to 512 columns and 7200 rows.  Since we were concerned only with speed tests and not prediction accuracy, the same data was used for training and testing.  

Subsets of the data of logarithmic size were used for speed tests.  The number of rows varied between 225 and 7200 with 512 columns, while the number of columns varied between 16 and 512 with 7200 rows.  Timing was done using Python's [timeit](https://docs.python.org/3.5/library/timeit.html) module with *number=10* and *repeat* at least 3 and scaled logarithmically.  Tests were run on a MacBook Air with dual core 1.4GHz Intel i5 processors.

#### Results
Several different algorithms were tested.
+ Logistic Regression
+ Naive Bayes
+ Support Vector Machines
+ Linear Support Vector Machines

<img src="https://github.com/bfetler/sklearn_clf_speed/blob/master/speed_test_plots/clf_time_row_fit.png" alt="clf fit rows" />

<img src="https://github.com/bfetler/sklearn_clf_speed/blob/master/speed_test_plots/clf_time_column_fit.png" alt="clf fit columns" />

<img src="https://github.com/bfetler/sklearn_clf_speed/blob/master/speed_test_plots/clf_time_row_predict.png" alt="clf predict rows" />

<img src="https://github.com/bfetler/sklearn_clf_speed/blob/master/speed_test_plots/clf_time_column_predict.png" alt="clf predict columns" />

