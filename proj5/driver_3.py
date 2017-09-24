#import csv
import pandas as pd
import os
import re
#from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import numpy as np

train_path = "./aclImdb/train/" # use terminal to ls files under this directory
test_path = "./imdb_te.csv" # test data for grade evaluation

def tokens(doc):
    return (tok.lower() for tok in re.findall(r"\w+", doc))

def token_freqs(doc):
    freq = defaultdict(int)
    for tok in tokens(doc):
        freq[tok] += 1
    return freq

def imdb_data_preprocess(inpath, stridelen, outpath="./", name="imdb_tr.csv", mix=False):
    #strip the new lines
    stopwords = [line.rstrip('\n') for line in open("stopwords.en.txt","r")]
    df = pd.DataFrame(columns = ["row", "text", "polarity"])
    row = 0
    filenum = 0
    posdirectory = os.fsencode(inpath+'pos')
    for file in os.listdir(posdirectory):
        filename = os.fsdecode(file)
        filenum +=1
        if filename.endswith(".txt") and filenum%stridelen==0 :#for now just do 10s
            
            f = open(inpath+'pos/'+filename,'r', encoding='latin1')
            
            reviewText = [line.split() for line in f.readlines()]
            f.closed
            reviewText = sum(reviewText,[])
            filtered_review = [word for word in reviewText if word not in stopwords]
        
            df.loc[row] = [row, ' '.join(filtered_review), 1]
            row+=1
            #print(row)
            continue

    negdirectory = os.fsencode(inpath+'neg')
    for file in os.listdir(negdirectory):
        filename = os.fsdecode(file)
        filenum +=1
        if filename.endswith(".txt") and filenum%stridelen == 0:#for now just do 10s
            

            f = open(inpath+'neg/'+filename,'r', encoding='latin1')
            reviewText = [line.split() for line in f.readlines()]
            f.closed
            
            reviewText = sum(reviewText,[])
            filtered_review = [word for word in reviewText if word not in stopwords]

            df.loc[row] = [row, ' '.join(filtered_review), 0]
            row+=1
            #print(row)
            continue

    df.to_csv(name)


def SGDUnigram(trainer = 'imdb_tr.csv'):
    stopwords = [line.rstrip('\n') for line in open("stopwords.en.txt","r")]
    
    trainingSet = pd.read_csv(trainer, encoding='latin1')
    trainXtext = trainingSet['text'].as_matrix()
    
    y =  list(trainingSet['polarity'].as_matrix())
##    print(trainXtext)
    vec = CountVectorizer(strip_accents='unicode', stop_words='english')
    trainX = vec.fit_transform(trainXtext,y)
##    trainX = (X.toarray())
    
    
    testSet = pd.read_csv(test_path, encoding='latin1')
    testXtext = testSet['text'].as_matrix()
   
    testX = vec.transform(testXtext)

    SGDCuni = SGDClassifier(loss="hinge", penalty="l1")
    
    SGDCuni.fit(trainX, y)
  
    print('writing data')
    outfile = open('unigram.output.txt', 'w') 
    
    for X in testX:
       results = SGDCuni.predict(X)
       outfile.write(str(int(results)).strip("[]")+'\n')
    outfile.close()
    

##    do cross reference check see if it is ok... so far only 0.7%
    X_train, X_test, y_train, y_test = train_test_split(trainX, y, test_size=0.4, random_state=0)
    SGDCuni = SGDClassifier(loss="hinge", penalty="l1")    
    SGDCuni.fit(X_train, y_train)  
    results = SGDCuni.predict(X_test)
    
    scores = cross_val_score(SGDCuni, X_test, y_test, cv=5)
    
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
 


def SGDBigram(trainer = 'imdb_tr.csv'):
    trainingSet = pd.read_csv(trainer, encoding='latin1')
    trainXtext = trainingSet['text'].as_matrix()    

    y =  list(trainingSet['polarity'].as_matrix())
    vec = CountVectorizer(ngram_range=(1, 2), strip_accents='unicode', stop_words='english')
    trainX = vec.fit_transform(trainXtext,y)
##    trainX = (X.toarray())

    testSet = pd.read_csv(test_path, encoding='latin1')
    testXtext = testSet['text'].as_matrix()
    
    testX = vec.transform(testXtext)
     
    SGDCBi = SGDClassifier(loss="hinge", penalty="l1")
    
    SGDCBi.fit(trainX, y)
  
    print('writing data')
    outfile = open('bigram.output.txt', 'w') 
    ##print(np.unique(y))
    for X in testX:
        results = SGDCBi.predict(X)
        outfile.write(str(int(results)).strip("[]")+'\n')
    outfile.close()
    
    
    ##    do cross reference check see if it is ok... so far only 0.7%
    X_train, X_test, y_train, y_test = train_test_split(trainX, y, test_size=0.4, random_state=0)
    SGDCuni = SGDClassifier(loss="hinge", penalty="l1")    
    SGDCuni.fit(X_train, y_train)  
    results = SGDCuni.predict(X_test)
    
    scores = cross_val_score(SGDCuni, X_test, y_test, cv=5)
    
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
 

    
def SGDUnigramTfidf(trainer = 'imdb_tr.csv'):
    trainingSet = pd.read_csv(trainer, encoding='latin1')
    trainXtext = trainingSet['text'].as_matrix()    

##    vec = CountVectorizer(strip_accents='unicode')
    vec = TfidfVectorizer(strip_accents='unicode', smooth_idf =False, norm = 'l2', stop_words='english') #(min_df=1)
    
    y =  list(trainingSet['polarity'].as_matrix())
    trainX = vec.fit_transform(trainXtext,y)
##    trainX = (X.toarray())
    

    testSet = pd.read_csv(test_path, encoding='latin1')
    testXtext = testSet['text'].as_matrix()
    
    testX = vec.transform(testXtext)
     

    SGDCuni = SGDClassifier(loss="hinge", penalty="l1")    
    SGDCuni.fit(trainX, y)
  
    print('writing data')
    outfile = open('unigramtfidf.output.txt', 'w') 
    
    for X in testX:
       results = SGDCuni.predict(X)
       outfile.write(str(int(results)).strip("[]")+'\n')
    outfile.close()
    
    ##    do cross reference check see if it is ok... so far only 0.7%
    X_train, X_test, y_train, y_test = train_test_split(trainX, y, test_size=0.4, random_state=0)
    SGDCuni = SGDClassifier(loss="hinge", penalty="l1", average = 15)    
    SGDCuni.fit(X_train, y_train)  
    results = SGDCuni.predict(X_test)
    
    scores = cross_val_score(SGDCuni, X_test, y_test, cv=5)
    
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
 


def SGDBigramTfidf(trainer = 'imdb_tr.csv'):
    trainingSet = pd.read_csv(trainer, encoding='latin1')
    trainXtext = trainingSet['text'].as_matrix()    
    y =  list(trainingSet['polarity'].as_matrix())
##    vec = CountVectorizer(ngram_range=(2, 2), strip_accents='unicode')
    vec = TfidfVectorizer(ngram_range=(1, 2),strip_accents='unicode', min_df=1) #(min_df=1)
    trainX = vec.fit_transform(trainXtext,y)
##    trainX = (X.toarray())
    
    

    
    testSet = pd.read_csv(test_path, encoding='latin1')
    testXtext = testSet['text'].as_matrix()
    
    testX = vec.transform(testXtext)
     
    SGDCBi = SGDClassifier(loss="hinge", penalty="l1")
    
    SGDCBi.fit(trainX, y)
  
    print('writing data')
    outfile = open('bigram.output.txt', 'w') 
    ##print(np.unique(y))
    for X in testX:
        results = SGDCBi.predict(X)
        outfile.write(str(int(results)).strip("[]")+'\n')
    outfile.close()
    
    ##    do cross reference check see if it is ok... so far only 0.7%
    X_train, X_test, y_train, y_test = train_test_split(trainX, y, test_size=0.4, random_state=0)
    SGDCuni = SGDClassifier(loss="hinge", penalty="l1")    
    SGDCuni.fit(X_train, y_train)  
    results = SGDCuni.predict(X_test)
    
    scores = cross_val_score(SGDCuni, X_test, y_test, cv=5)
    
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
 

    
if __name__ == "__main__":
##    create the training set.
    imdb_data_preprocess(train_path, 1) #uncomment when submitting
    print("done preprocessing")
    SGDUnigram()
    SGDUnigramTfidf()

    SGDBigram()
    SGDBigramTfidf()    
    

##    '''train a SGD classifier using unigram representation,
##    predict sentiments on imdb_te.csv, and write output to
##    unigram.output.txt'''
##  	
##    '''train a SGD classifier using bigram representation,
##    predict sentiments on imdb_te.csv, and write output to
##    bigram.output.txt'''
##     
##     '''train a SGD classifier using unigram representation
##     with tf-idf, predict sentiments on imdb_te.csv, and write 
##     output to unigramtfidf.output.txt'''
##  	
##     '''train a SGD classifier using bigram representation
##     with tf-idf, predict sentiments on imdb_te.csv, and write 
##     output to bigramtfidf.output.txt'''
##     pass



