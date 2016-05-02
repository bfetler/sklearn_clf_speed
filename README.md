## Scikit-learn classifier speed tests

The [Scikit-learn](http://scikit-learn.org/stable/index.html) Python library provides an API for an easy-to-use set of packages for Machine Learning, including classification, regression, clustering and dimensionality reduction.  In comparing different methods, one needs to consider many factors:

+ The problem you're trying to solve.
+ The appropriateness of the method for your data.
+ The speed of the algorithm.

#### Classification
Classification is generally suited to problems where you have a number of independent variables *X*, and two or more classes in a target variable *y*.  The general problem is to predict *y* given *X*.  

Data is usually split into a training data set to fit the model, and a test data set to test how well the model and algorithm performs on unknown data.  If the test data contains the target *y* as well as independent *X's*, one may evaluate the model accuracy as to whether or not it predicted the correct values.

The Scikit-learn classifier API has a number of common methods for a classifier instance *clf*, including:
+ *clf.fit()* : applied to a training data set, fits model
+ *clf.predict()* : applied to a test data set, predicts *y* given *X*

#### Speed Tests
Speed tests were performed on the *fit()* and *predict()* methods for several classifiers.  The [Human Activity by Smartphone Dataset]() was chosen since it has 561 columns and 7352 rows.  The data was pared down to 512 columns and 7200 rows, and subsets of the data with logarithmic size were used for speed tests.  Timing was done using Python's [timeit]() module.

#### Results
Several different algorithms were tested.
+ Logistic Regression
+ Naive Bayes
+ Support Vector Machines
+ Linear Support Vector Machines

Fitting of rows.

Fitting of columns.

Predicting rows.

Predicting columns.

