#percepton learning algorithm
import numpy as np
##import matplotlib.pyplot as plt

import csv
##import pandas as pd

import sys

print("This is the name of the script: ", sys.argv[0])
print("Number of arguments: ", len(sys.argv))
print("The arguments are: " , str(sys.argv))

print(str(sys.argv[1]))
print(str(sys.argv[2]))

X = []
y = []
with open(sys.argv[1], newline='') as csvfile:
     inputcsv = csv.reader(csvfile, delimiter=',', quotechar='|')
     for row in inputcsv:
         rowInt = []
         for elem in row:
             int1 = int(elem)
             rowInt.append(int1)
         X.append(rowInt[0:2])
         y.append(rowInt[2])
         
X = np.array(X)
y = np.array(y)


    
    
    
    

print(X)
print(y)

w = np.array([0, 0])
b = 0
tol= 10
yj = np.array([])
done = False
weights = []

while done == False:
    yj = []
    i = 0
    done = True
    for xj in X:
        fxi = -1
        if w.dot(xj)+b>0: # need to actual do perceptron output
             fxi = 1
        
        
        
        if y[i]*fxi <=0 :
            done = False
            #yj = w.dot(xj)+b #actual out
            w[0]= w[0]+y[i]*xj[0]
            w[1]= w[1]+y[i]*xj[1]
            b = b+y[i]
        
        i += 1
##    with open(sys.argv[2], 'w', newline='') as csvfile:
##        outputweights = csv.writer(csvfile, delimiter=',',
##                            quotechar='|', quoting=csv.QUOTE_MINIMAL)      
##        outputweights.writerow([w[0],w[1],b])
    print('w = ', w)
    print('b = ', b)
    weights.append([w[0],w[1],b])
    
    
with open(sys.argv[2], 'w', newline='') as csvfile:
    outputweights = csv.writer(csvfile, delimiter=',',
                  quotechar='|', quoting=csv.QUOTE_MINIMAL)      
    for rows in weights:
        outputweights.writerow([rows[0],rows[1],rows[2]])

##fig, ax = plt.subplots()
##
##ax.scatter(X[:,0], X[:,1], c=y)
##
##
##    
####print(X_train[:,0])
##ax.set_xlabel(r'$X_0$', fontsize=15)
##ax.set_ylabel(r'$X_1$', fontsize=15)
##
##x00 = 0
##x10 = -b/w[1]
##print(x10)
##
##x01 = 15
##x11 = -b/w[1] - w[0]*15/w[1]
##print(x11)
##plt.plot([x00, x01], [x10, x11], 'k-')
##
##
##ax.grid(True)
##fig.tight_layout()
##
##plt.show()
