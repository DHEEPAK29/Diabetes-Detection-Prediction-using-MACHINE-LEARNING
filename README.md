# DIABETES-DETECTION-USING-MACHINE-LEARNING
 
 DIABETES DETECTION USING MACHINE LEARNING

Description: Main objective of this project is to estimate the possibilities of people  becoming a diabetic.

Technologies used: Machine Learning

Software and application used :   Jupyter Notebook,  Python 3.9.6.

ML Libraries used: 	Pandas , NumPy  , Matplotlib ,Seaborn.

# SYNOPSIS 
INTRODUCTION
OBJECTIVE
ANALYSIS CRITERIA
LIBRARIES USED
ALGORITHM
FUNCTIONS –DEFINITIONS
INFERENCE
REFERENCE


# INTRODUCTION
What is Diabetes?
Diabetes is a metabolic disease that occurs when your blood glucose, also called blood sugar, is too high.
Type I diabetes: Also known as juvenile diabetes, this type occurs when the body fails to produce insulin. People with type I diabetes are insulin-dependent, which means they must take artificial insulin daily to stay alive.
Type 2 diabetes: Type 2 diabetes affects the way the body uses insulin. While the body still makes insulin, unlike in type I, the cells in the body do not respond to it as effectively as they once did. This is the most common type of diabetes, according to the National Institute of Diabetes and Digestive and Kidney Diseases, and it has strong links with obesity.
Gestational diabetes: This type occurs in women during pregnancy when the body can become less sensitive to insulin. Gestational diabetes does not occur in all women and usually resolves after giving birth.


# OBJECTIVE
 The objective of this project is to predict and match with Machine Learning model in comparison to real time data based on certain diagnostic parameters included in the dataset.
 Pima Indians with type 2 diabetes are metabolically characterized by, <br>
	Obesity.<br>
	Insulin resistance.<br>
	Insulin secretory dysfunction. <br>
	Increased rates of endogenous glucose production. <br>
which are the clinical characteristics that define this disease across most populations . 
Using mathematical models to analyze how Pregnant women are prone to be Diabetic.
Generate Clear understanding of how the disease is vulnerable among population.


# ANALYSIS CRITERIA
Pregnancies-Number of times a women was pregnant.

Glucose-Plasma glucose concentration a 2 hours in an oral glucose tolerance test.

Blood Pressure-Diastolic blood pressure (mm Hg)

Skin Thickness-Triceps skin fold thickness (mm)

Insulin-2-Hour serum insulin (mu U/ml)

BMI-Body mass index (weight in kg/(height in m)^2)

Diabetes Pedigree Function-Diabetes pedigree function(Pedigree Tree).

Age-Age of the women(in years)

Outcome-Class variable (0 or 1) 268 of 768 are 1, the others are 0

# LIBRARIES USED
SEABORN  -For data visualization .Boxplot .
MATPLOTLIB  -2D/3D plotting .
PANDAS – import comma separated value from csv and Data manipulation.
NUMPY- make light weight arrays and math functions

# ALGORITHM
1.Import Libraries.
2.import dataset from .csv file using pandas read_csv function.
3.Display info() to analyze the data.
4.Group by positivity rate from the given data.
5.Use pyplot hist()function to represent each criteria graphically.
6.Use Boxplot() from seaborn to statistically represent Each criteria of analytics .
7.Use pandas Values method to return view Objects as array.
8. from sklearn.model_selection import train_test_split
9. 
10.from sklearn.tree import DecisionTreeClassifier
11.from sklearn.ensemble import RandomForestClassifier
12.Use fit() function to adjust weights according to data values so that better accuracy can 	be achieved. 
13.Randomforest prediction result vs decision tree prediction results using classification_report(x,y).

# FUNCTIONS –DEFINITIONS
1.HISTOGRAM
Syntax: matplotlib.pyplot.hist(size)
Return:This returns the list of individual patches used to create the histogram.

![image]( )
<img src="https://user-images.githubusercontent.com/83426515/165886489-5a52fdd9-441e-4617-ba11-8cb6294830f1.png" width="50%" height="50%">

2.BOX PLOT
Syntax: seaborn.boxplot(x=None, y=None, data=None)
Returns: It returns the Axes object with the plot drawn onto it. 
Type used:  Draw a single horizontal box plot using only one axis.
A box plot helps to maintain the distribution of quantitative data in such a way that it facilitates the comparisons between variables or across levels of a categorical variable.
 The main body of the box plot showing the quartiles and the median’s confidence intervals if enabled.

![image]()
<img src="https://user-images.githubusercontent.com/83426515/165886540-e5f1fd8d-d007-4d81-a508-80dbc9d05adb.png" width="50%" height="50%">

3.VALUES
Syntax: DataFrame.values
Parameter : None
Returns : array
Pandas DataFrame is a two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns). 
Arithmetic operations align on both row and column labels.
 It can be thought of as a dict-like container for Series objects. This is the primary data structure of the Pandas.

![image]()
<img src="https://user-images.githubusercontent.com/83426515/165886569-efda1da3-8f0c-4190-a245-98b9e38702f3.png" width="50%" height="50%">
4.sklearn.model_selection import train_test_split
Syntax: sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None).

The train-test split is a technique for evaluating the performance of a machine learning algorithm. 
It can be used for classification or regression problems and can be used for any supervised learning algorithm. 
Purpose: This means that you can’t evaluate the predictive performance of a model with the same data you used for training. 
You need evaluate the model with fresh data that hasn’t been seen by the model before. You can accomplish that by splitting your dataset before you use it.

5.sklearn.ensemble.RandomForestClassifier
A random forest is an estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 
class sklearn.ensemble. RandomForestClassifier()
Used function::	predict(X) --Predict class for X.

6.sklearn.tree.  DecisionTreeClassifier
Used::fit(X, y[, sample_weight, check_input, …])-Build a decision tree classifier from the training set (X, y).
Use fit() function to adjust weights according to data values so that better accuracy can be achieved. 

7.sklearn.metrics.classification_report
Build a text report showing the main classification metrics.
Return type:
 precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
F1 score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.
The F-beta score weights recall more than precision by a factor of beta. beta == 1.0 means recall and precision are equally important.
support is the number of occurrences of each class in y_true.

macro average (averaging the unweighted mean per label)
weighted average (averaging the support-weighted mean per label)
sample average (only for multilabel classification). 
Micro average (averaging the total true positives, false negatives and false positives)

# INFERENCE
CASE A::
Prediction BY - DecisionTreeClassifier()

![image]()
<img src="https://user-images.githubusercontent.com/83426515/165886740-73b3c98c-2904-4bbf-8e42-3ee00c2e558c.png" width="50%" height="50%">
<img src="https://user-images.githubusercontent.com/83426515/165886744-751f01ba-e8c0-4c6b-9065-8e8e5f44ee1f.png" width="50%" height="50%">
![image]()

CASE B::
Prediction  BY – RandomForest()
![image]()
<img src="https://user-images.githubusercontent.com/83426515/165886776-ad6ed707-48fa-4543-82e3-1fb3177758c6.png" width="50%" height="50%">

# REFERENCE
KAGGLE-Pima Indians Diabetes Database-Predict the onset of diabetes based on diagnostic measures.<br>
scikit-learn: machine learning in Python — scikit-learn 0.24.2 documentation.<br>
Dataset: National Institute of Diabetes and Digestive and Kidney Diseases. 



