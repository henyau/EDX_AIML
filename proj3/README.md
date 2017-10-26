# **Some classification/regression algorithms** 

The file `problem1_3.py` contains a simple implementation of the percepton learning algorithm for classification
Usage: `python problem1_3.py input1.csv output1.csv`
where input1.csv is has three columns, x,y and a label of either +1 or -1.

---
The file `problem2_3.py` demonstrates a simple implementation of the 
gradient descent algorithm for regression

Usage: `python problem2_3.py input2.csv output2.csv`

where input2.csv is has three columns of real numbers, x1,x2,y

output2.csv has five columns, alpha, number of iterations, and coefficients
beta such that beta = (X'X)^-1 X'y

alpha is the learning rate which prescribed in alpha_vec
---

The file `problem3_3.py` demonstrates using sklearn using support vector machine (SVM) 
classifier using different kernels and other classifiers

Usage: python problem3_3.py input3.csv 

where input3.csv has three columns, real data A and B and a label of 0 or 1
---






