## Scikit-learn classifier speed tests

The [Scikit-learn](http://scikit-learn.org/stable/index.html) Python library provides an API and easy-to-use set of packages for classification and regression in Machine Learning.  In comparing different methods, one needs to consider many factors:

+ The problem you're trying to solve.
+ The appropriateness of the method for your data.
+ The speed of the algorithm.

Classification is generally suited to problems where you have a number of independent variables *X*, and two or more classes in a target variable *y*.  The general problem is to predict *y* given *X*.

The Scikit-learn classifier API has a number of common methods for a classifier instance *clf*, including
+ *clf.fit()* : applied to a training data set
+ *clf.predict()* : applied to a test data set

