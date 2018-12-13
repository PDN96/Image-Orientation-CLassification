#!/usr/bin/env python3

'''
Data Handling
1.Checked for missing values in the data sets, if there are missing values, we can fill missing values by mean or median for the numeric features, and by mode for categorical values.
In the given data sets, we didn’t find any missing value.
2.Imported data files as Data Frame and converted the Data Frame to into Numpy Array for more convenience of using data and conversion in Numpy Array helped to decrease the time of execution.
Comments for KNN
Steps for Implementation
1.	Defined a function (def distance(x,y)) to calculate Euclidean Distance. Here, for each test data, this function calculates the distance with each training training data.
2.	Defined a function (Function name: def knn(test_data,train_data,k)) to find the nearest neighbors by finding their index. The function takes train data , test data and k value as input and returns index of the k nearest neighbors as output.
3.	Defined a function (Function name: def find_max_mode(list_data)) to find mode of the labels of nearest neighbors (voting on correct orientation). This function takes labels of k nearest neighbors as input and produces output of the label having maximum frequency.
4.	Defined a prediction function (Function name: def prediction (a,b,train_label, k))  to predict the labels of the given test data. Here a – train data, b-test data, k – number of nearest neighbors). 
5.	Defined a function (Function name: def accuracy(predicted,actual)) to calculate accuracy of the model by comparing the predicted labels and actual test labels.
6.	Calculated by maximum accuracy by tuning the hyperparameter (k – number of nearest neighbors) and plotted the graph of K-value vs Accuracy


Comments for Random Forest
Implementation of random forest:
1.	The main component of a random forest classifier is the decision tree.
2.	In order to implement a decision we split iteratively using a greedy algorithm. At each node we find the best attribute to split on.
3.	We handle the continuous splitting criteria by setting the threshold to half the value of the highest pixel. (128).
4.	In order to the select the best splitting attribute we calculate the information gain of the split. Whichever split has the maximum information gain is selected. 
5.	Now, we create a forest of such decision trees. Each tree is trained on a subset of the training data selected randomly.
6.	On testing each decision tree predicts and then the final decision is made by selecting the class which has the maximum votes.

Comments for Adaboost
To solve the multi-class problem, we converted this into a one vs all problem. That is (0 vs 90), (0 vs 180), (0 vs 270), (90 vs 180), (90 vs 270), and (180 vs 270)
This generated 6 sets of training data.
 For each training set, we take a majority vote. If majority vote for a class then that class is assigned Positive otherwise Negative
Initially the weights are initialized as 1/N. Whenever we get the correct answer we decrease the initialized weights otherwise we increase the weights of the incorrectly classified example
. Normalization is then done. 	+Initially the weights are initialized as 1/N. 
Whenever we get the correct answer we decrease the initialized weights otherwise we increase the weights of the incorrectly classified example. 
Increase is done by adding the error,
+Normalization is then done.
 An accuracy ranging from 68-71% was achieved. 
Even if we increased the number of random pixel pairs compared, the accuracy achieved didn’t increase by more than 71% but at a much higher computation cost.
An accuracy ranging from 68-71% was achieved. Even if we increased the number of random pixel pairs compared, the accuracy achieved didn’t increase by more than 71% but at a much higher computation cost. 
'''


import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from collections import Counter
import operator
import pickle
import sys
from sklearn.model_selection import train_test_split
import random
import itertools

#importing file 
data_train = pd.read_table(sys.argv[2],delim_whitespace=True,header=None)
data_test = pd.read_table(sys.argv[2], delim_whitespace=True,header=None)

#Created a separate dataframe for labels of training data 
train_data_label=data_train[1]

#Converted the train labels from dataframe to array
train_data_label=train_data_label.values

#Dropping the columns of labels from train data
data_train.drop(0,axis=1,inplace=True)
data_train.drop(1,axis=1,inplace=True)

#crated a separate dataframe for labels in test data
test_data_label=data_test[1]

#converted the test labels from dataframe to array
test_data_label=test_data_label.values

#Dropping the columns of lables from test data
data_test.drop(0,axis=1,inplace=True)
data_test.drop(1,axis=1,inplace=True)


#converting train and test data without labels to array
data_train=data_train.values
data_test=data_test.values


#Initialized Dictionories 
tree = {}
decision = {}
visited = {}

depth_n = 15
n_trees = 50

tree_list = []
decision_list = []

labels = [0, 90, 180, 270]





import itertools

# function to read data sets file for Adaboost
def read_file(fname):
    print("Reading data from", fname, "...")
    image = np.loadtxt(fname, usecols=0, dtype=str)
    X = np.loadtxt(fname, usecols=[i for i in range(2, 194)], dtype=int)
    y = np.loadtxt(fname, usecols=1, dtype=int)
    shuffle_indices = [i for i in range(len(y))]
    random.shuffle(shuffle_indices)
    X  = X[shuffle_indices, ]/255
    image = list(image[shuffle_indices,])
    y = y[shuffle_indices, ]

    return image, X, y

#Function to display output file for Adaboost
def to_file(image, pred):
    f = open('output.txt', 'w')
    for i in range(0, len(image)):
        f.write(image[i] + ' ' + str(pred[i]) + '\n')
    f.close()

#
def get_possible_pairs(n_features):
    possible_pairs = []
    for x in range(n_features):
        for y in range(x, n_features):
            if x!=y:
                possible_pairs.append((x,y))
    return possible_pairs


class AdaBoost(object):
    def __init__(self, k):
        self. k = k

    def train(self, X_train, y_train, variables):
        y_unique = list(set(y_train))
        vote_classifier = {}
        weights = np.array([float(1)/float(len(y_train))]*len(y_train))
        for v in range(len(variables)):
            error = 0
            index1 = variables[v][0]
            index2 = variables[v][1]
            decision_stump = []
            decision_stump_category = {'Negative':{},'Positive':{}}
            for i in range(0, len(X_train)):
                if X_train[i][index1] < X_train[i][index2]:
                    decision_stump.append('Negative')
                    if y_train[i] in decision_stump_category['Negative']:
                        decision_stump_category['Negative'][y_train[i]] += 1
                    else:
                        decision_stump_category['Negative'][y_train[i]] = 1

                else:
                    decision_stump.append('Positive')
                    if y_train[i] in decision_stump_category['Positive']:
                        decision_stump_category['Positive'][y_train[i]] += 1
                    else:
                        decision_stump_category['Positive'][y_train[i]] = 1

            m = 0
            for orient in decision_stump_category['Positive']:
                if decision_stump_category['Positive'][orient] > m:
                    m = decision_stump_category['Positive'][orient]
                    Positive_Class = orient
            m = 0
            for orient in decision_stump_category['Negative']:
                if decision_stump_category['Negative'][orient] > m:
                    m = decision_stump_category['Negative'][orient]
                    Negative_Class = orient

            decision_stump_classification = []
            for i in range(len(y_train)):
                if decision_stump[i] == 'Negative':
                    decision_stump_classification.append(Negative_Class)
                else:
                    decision_stump_classification.append(Positive_Class)
                if decision_stump_classification[i] == y_train[i]:
                    continue
                else:
                    error = error + weights[i]

            if error < 0.5:
                for i in range(0, len(y_train)):
                    if decision_stump_classification[i] == y_train[i]:
                        weights[i] = weights[i] * error / (1.0 - error)

                sum_weights = sum(weights)
                weights = weights / sum_weights
                vote_classifier[variables[v]] = {}
                vote_classifier[variables[v]]['weight'] = math.log((1 - error) / error)
                vote_classifier[variables[v]]['Positive_Class'] = Positive_Class
                vote_classifier[variables[v]]['Negative_Class'] = Negative_Class


        return vote_classifier, y_unique

    def test(self, X_test, y_test, vote_classifier, y_unique):
        y_unique_dict = {}
        y_unique_dict[y_unique[0]] = 1
        y_unique_dict[y_unique[1]] = -1
        classification = [0] * len(y_test)
        classifiers = vote_classifier
        for classifier in classifiers:
            index1 = classifier[0]
            index2 = classifier[1]
            for i in list(range(0,len(y_test))):
                if X_test[i][index1] < X_test[i][index2]:
                    vote = classifiers[classifier]['Negative_Class']
                    classification[i] += y_unique_dict[vote] * classifiers[classifier]['weight']
                else:
                    vote = classifiers[classifier]['Positive_Class']
                    classification[i] += y_unique_dict[vote] * classifiers[classifier]['weight']
        classification_category = []
        for i in range(0, len(y_test)):
            if classification[i] < 0:
                classification_category.append(y_unique[1])
            else:
                classification_category.append(y_unique[0])
        return classification_category

# Function for Euclidean distance calculation

def distance(x,y):   
    return np.sqrt(np.sum((x-y)**2))

        
# function to find nearest neighbours
import operator
def knn(test_data,train_data,k):
    dist=[]                        #dist- is a list which conatins neighbours and its corresponding distances from the test data
    length=len(train_data)
    for j in range(length):
        d=distance(train_data[j],test_data)
        dist.append((j,d))
    dist.sort(key=operator.itemgetter(1))
    neighbours=[]
    for x in range(k):
        neighbours.append(dist[x][0])
    return neighbours

    
#function to find the mode of the labels of the nearest neighbours
import statistics
def find_max_mode(list_data):
    list_table = statistics._counts(list_data)
    len_table = len(list_table)

    if len_table == 1:
        max_mode = statistics.mode(list_data)
    else:
        new_list = []
        for i in range(len_table):
            new_list.append(list_table[i][0])
        max_mode = max(new_list) 
    return max_mode

#Function to find predictions of test data by K-NN model
#k - Number of nearest neighbours 
#a - Input Train_data
#b - Test data
#train_label 

def prediction(a,b,train_label, k):
    predictions = []
    for i in range(len(b)):
        neigh=knn(b[i],a,k)
        n_labels = []                                    #n_labels is a list which contains labels of the predicted neighbours
        n_labels = train_label[neigh]
        predictions.append(find_max_mode(n_labels))      #voting on the correct orientation
    return predictions

#Function to find Accuracy 
#Accuracy was found by comparing the predicted value by model and actual value on the test data
def accuracy(predicted,actual):
        count1=0
        for i in range(len(predicted)):
            if predicted[i]==actual[i]:
                count1+=1
            accuracy1=count1/len(predicted)
        return(accuracy1)
 

#Defined a function to calculate the entropy for random forest, the function takes label and indexes as input and returns entropy as output       
def calc_entropy(label, indexes):
    label = label[indexes]
    c  = Counter(label)
    entropy = 0
    total_len = len(label)
    if(total_len == 0):
        return np.inf
    for y in labels:
        p = c[y]/total_len
        if(p != 0):
            entropy+= (-p)*np.log2(p)
    return entropy

#Calculation of Information for child nodes in random forest
def cal_info(feat, train_data, label_data):
    left = []
    right = []
    for i, val in enumerate(train_data[:,feat]):
        if(val> 128):
            right.append(i)
        else:
            left.append(i)
    term1 = calc_entropy(label_data, left)
    term2 = calc_entropy(label_data, right)
    return left, right, (len(left)/len(train_data))*term1 + (len(right)/len(train_data))*term2, term1,  term2

#Defined a function to find the split
def dt_split(train, label, index, depth):

    if(depth == depth_n):
        decision[np.floor(index/2)] = Counter(label).most_common(1)[0][0]
        return
    
    min_value = math.inf
    ind = -1
    l_index = []
    r_index = []
    l_val = 0
    r_split = 128
    r_val = 0
    col_len = np.shape(train)[1]
    for feat in range(col_len):
        left, right, value, left_val, right_val = cal_info(feat, train, label)
        if(value < min_value):
            min_value = value
            ind = feat
            l_index = left
            r_index = right
            l_val = left_val
            r_val = right_val
    if(ind == -1):
        return
    visited[ind] = True

    if(len(l_index) != 0):
        if(l_val < 0.4):
            decision[np.floor(index/2)] = label[l_index[0]]
        else:
            dt_split(train[l_index], label[l_index], index*2, depth+1)
    
    if(len(r_index) != 0):
        if(r_val < 0.4):
            decision[np.floor(index/2)] = label[r_index[0]]
        else:
            
            dt_split(train[r_index], label[r_index], index*2 + 1, depth+1)
    tree[index] = ind

#Defined a function to predict the label of the test data in random forest model
def test_rv(test_x, test_y):
    predicted = []
    file = open('forest_model.txt', 'rb')
    decision_list = pickle.load(file)
    tree_list = pickle.load(file)

    for i in range(len(test_x)):
        res = {0:0, 90:0, 180:0, 270:0}
        for tree_index in range(n_trees):
            split_index = 1
            for de in range(depth_n):
                if(split_index in decision_list[tree_index]):
                    
                    res[decision_list[tree_index][split_index]] +=1
                    break
                ll = tree_list[tree_index][split_index]
                if (test_x[i][ll] < 128) :
                    split_index = split_index*2
                else:
                    split_index = split_index*2 + 1
        sorted_x = sorted(res.items(), key=operator.itemgetter(1))
        predicted.append(max(res, key=res.get))

    return predicted

#Defined a function to train random forest model
def rv_train():
    global tree 
    global decision 
    global visited

    global tree_list
    global decision_list
    for i in range(n_trees):
        X_train, X_test, y_train, y_test = train_test_split(data_train, train_data_label, test_size=0.33)

        dt_split(X_train, y_train, 1, 0)
        tree_list.append(tree)
        decision_list.append(decision)
        tree = {}
        decision = {}
        visited = {}

    file = open('forest_model.txt','wb')
    pickle.dump(decision_list, file)
    pickle.dump(tree_list, file)
    file.close()


if(sys.argv[1] == "test"):
    predicted = []
    #print("test")

    if(sys.argv[4] == "nearest"):
        #print("nearest")
        data_train1 = pd.read_table("nearest_model.txt", delim_whitespace=True,header=None)

        train_data_label = data_train1[1]
        train_data_label = train_data_label.values
        data_train1.drop(0,axis=1,inplace=True)
        data_train1.drop(1,axis=1,inplace=True)
        data_train = data_train1.values
        predicted = prediction(data_train,data_test, train_data_label, 41)

    if(sys.argv[4] == "forest"):
        #print("done")
        predicted = test_rv(data_test, test_data_label)
    
    if((sys.argv[4] == "adaboost") or (sys.argv[4] == "best")):
        #print("here")
        task, fname, model_file, model = sys.argv[1:]
        image, X, y = read_file(fname)
        models = pickle.load(open(model_file, "rb"))
        n_labels = len(set(y))
        num_variables = int(n_labels * (n_labels - 1) / 2)
        n_features = len(X[0])

        vote_classifier_y_unique, adaboost = models
        classification_category = []
        for i in range(num_variables):
            classification_category.append(
                adaboost.test(X, y, vote_classifier_y_unique[i][0], vote_classifier_y_unique[i][1]))

        final_classification = []
        count_correct = 0

        for i in range(len(y)):
            lst = []
            for pair in range(num_variables):
                lst += [classification_category[pair][i]]
            final_classification.append(max(lst, key=lst.count))
            if final_classification[i] != y[i]:
                continue
            else:
                count_correct += 1

        score = round(float(count_correct) / float(len(y)), 2) * 100

        print("Writing to a file...")
        to_file(image, final_classification)
        print("Accuracy", score, "%")


    if(sys.argv[4] == "nearest" or sys.argv[4] == "forest" ):
        acc=(accuracy(predicted,test_data_label))*100
        print(acc)
        # code to display the output file in the required format
        labels1 = pd.read_csv(sys.argv[2],delim_whitespace=True,header=None).iloc[:,0]
        file=open("output.txt", "w")
        for i in range(len(data_test)):
            file.write(labels1[i]+ " ")
            file.write(str(predicted[i]) + '\n')



        

if(sys.argv[1] == "train"):
    #print("train")
    if(sys.argv[4] == "nearest"):
        pass
    if(sys.argv[4] == "forest"):
        rv_train()
    if(sys.argv[4] == "adaboost" or sys.argv[4]=="best"):
        # call test function and return list of predicted values.
        task, fname, model_file, model = sys.argv[1:]
        image, X, y = read_file(fname)
        
        k = 2000
        n_features = len(X[1])
        possible_pairs = get_possible_pairs(n_features)
        n_labels = len(set(y))
        num_variables = int(n_labels * (n_labels - 1) / 2)
        variables = []
        for i in range(num_variables):
            variables.append(random.sample(possible_pairs, k))
        y_trains = {}
        X_trains = {}
        for label in set(y):
            y_trains[label] = []
            X_trains[label] = []
        for i, label in enumerate(y):
            y_trains[label].append(y[i])
            X_trains[label].append(X[i])

        X_train_pairs = []
        y_train_pairs = []
        for c1, c2 in list(itertools.combinations(set(y), 2)):
            X_train_pairs.append(X_trains[c1] + X_trains[c2])
            y_train_pairs.append(y_trains[c1] + y_trains[c2])

        adaboost = AdaBoost(k)

        vote_classifier_y_unique = []
        for i in range(num_variables):
            vote_classifier_y_unique.append(adaboost.train(X_train_pairs[i], y_train_pairs[i], variables[i]))

        models = (vote_classifier_y_unique, adaboost)
        pickle.dump(models, open(model_file, "wb"), protocol=2)
