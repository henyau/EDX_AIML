import numpy as np
import csv
import sys

##print("This is the name of the script: ", sys.argv[0])
##print("Number of arguments: ", len(sys.argv))
##print("The arguments are: " , str(sys.argv))

##print(str(sys.argv[1]))
##print(str(sys.argv[2]))

X = []
y = []
with open(sys.argv[1], newline='') as csvfile:
     inputcsv = csv.reader(csvfile, delimiter=',', quotechar='|')
     for row in inputcsv:
         rowInt = []
         for elem in row:
             float1 = float(elem)
             rowInt.append(float1)
         X.append(rowInt[0:2])
         y.append(([rowInt[2]]))
         
X = np.array(X)
y = np.array(y)

stdx0 = np.std(X[:,0])
stdx1 = np.std(X[:,1])
stdy = np.std(y)

meanx0 = np.mean(X[:,0])
meanx1= np.mean(X[:,1])
##meany = np.std(y)
n = len(y)

X[:,0] = (X[:,0]-meanx0)/stdx0
X[:,1] = (X[:,1]-meanx1)/stdx1

ones_col = np.ones((n,1))
X = np.hstack((ones_col,X ))


##print(X)
##print(Xscaled1)

alpha_vec = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 2]
maxIter = 100



result_Mat = []

for alpha in alpha_vec:
    beta_vec = np.zeros((3,1))
    for i in range (0, maxIter):
        beta_vec[0] = beta_vec[0]- (alpha/n)*(X.dot(beta_vec)-y).sum()
        
##        Xtemp = np.array((X[:,1]))
##        Xtemp = Xtemp[np.newaxis, :]        
##        beta_vec[1] = beta_vec[1]- (alpha/n)*((X.dot(beta_vec)-y).dot(Xtemp)).sum()
##        
##        Xtemp = np.array((X[:,2]))
##        Xtemp = Xtemp[np.newaxis, :]        
##        beta_vec[2] = beta_vec[2]- (alpha/n)*((X.dot(beta_vec)-y).dot(Xtemp)).sum()
##        Xtemp = np.array((X[:,1:2]))
        sum2 = 0
        for j in range (0,n):
            roww = np.array([X[j][1], X[j][2]])
            roww = roww[np.newaxis, :]  
            sum2 += (X[j][:].dot(beta_vec)-y[j]).dot(roww)
       
        sum2 = sum2[:,np.newaxis]              
        
        beta_vec[1:3] = beta_vec[1:3]- (alpha/n)*sum2
        
        
    
    
    result_Mat.append([alpha, maxIter, float(beta_vec[0]), float(beta_vec[1]), float(beta_vec[2])])
##print(result_Mat)
with open(sys.argv[2], 'w', newline='') as csvfile:
    result_csv = csv.writer(csvfile, delimiter=',',
                  quotechar='|', quoting=csv.QUOTE_MINIMAL)      
    for rows in result_Mat:
        result_csv.writerow([rows[0],rows[1],rows[2],rows[3],rows[4]])

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

