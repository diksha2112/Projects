__author__ = 'yatinsharma'
__author__ = 'yatinsharma'

import math
import pandas as pd
import numpy as np
import random
from statistics import mode


#function to read file
def read_data(floc):
    import csv

    with open(floc, 'r') as f:
        reader = csv.reader(f)
        list_data = list(reader)
        list_data.__delitem__(0)
        return list_data

#function to calculate entropy
def entropy(data):
    val_freq = {}
    data_entropy = 0.0
    subset_entropy = 0.0
    for record in data:
        #storing the count of each label in dictionary
        if (record[0] in val_freq):
            val_freq[record[0]] += 1.0
        else:
            val_freq[record[0]] = 1.0

    # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq / len(data)) * math.log(freq / len(data), 2)
    return data_entropy




#function to return the current best feature to split on by calculating the info gain
def best_feature(rows):  #rows is the set, either whole dataset or part of it in the recursive call,
    t = rows
    orig_entropy = entropy(t)
    best_gain = 0.0
    best_attribute = (0, 0)
    best_set = None
    best_subset1 = []
    best_subset2 = []
    subset_entropy = 0.0
    column_count = len(t[0])
    # Calculate the frequency of each of the values in the target attribute
    for col in range(1, column_count):
        val_freq = {}
        for record in t:
            if (record[col] in val_freq):
                val_freq[record[col]] += 1.0
            else:
                val_freq[record[col]] = 1.0

        for val in val_freq:

            weight = val_freq[val] / sum(val_freq.values())
            #spliting data into left and right based upon the feature value
            subset1 = [record for record in t if record[col] == val]  #can get caught , change thisS
            subset2 = [record for record in t if record[col] != val]
            subset_entropy = weight * entropy(subset1) + (1 - weight) * entropy(subset2)
            info_gain = orig_entropy - subset_entropy
            #choosing the feature based upon the highest info gain
            if info_gain > best_gain and len(subset1) > 0 and len(subset2) > 0:
                best_gain = info_gain
                best_attribute = (col, val)
                best_subset1 = subset1
                best_subset2 = subset2

    return(best_attribute,best_subset1,best_subset2)


#function to create leaf node
def create_leaf(target_values):
    lable = {}
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True,
            'split_feature_value':None    }

    # Count the number of data points that are +1 and -1 in this node.
    num_ones=0
    num_zeros=0
    if len(target_values)==0:
        pass
    for row in target_values:
        if(row not in lable):
            lable[row] = 1
        else:
            lable[row] +=1

    max = 0
    prediction = None
    for val in lable:
        if(lable[val]>max):
            max=lable[val]
            prediction = val

    leaf['prediction'] = prediction

    return leaf


#function to create decision tree
def decision_tree_create(data, max_depth,current_depth = 0):
    leftsplit = []
    rightsplit=[]
    #creatting leaf if entropy is zero
    if entropy(data)==0:
        return create_leaf([record[0] for record in data])

    #creating leaf if max depth reached
    if current_depth >= max_depth:
        return create_leaf([record[0] for record in data])
#finding the best feature
    split_feature,leftsplit,rightsplit =best_feature(data)

    if(len(leftsplit)==0 and len(rightsplit)==0):
        #creating leaf node if no further split possible
        return create_leaf([record[0] for record in data])
    if(len(leftsplit))==0:
        # creating leaf node if left split  is empty
        return create_leaf([record[0] for record in rightsplit])
    if(len(rightsplit)==0):
        # creating a leaf node if right node is empty
        return create_leaf([record[0] for record in leftsplit])


#create a leaf node ,if split is perfect
    if len(leftsplit)==len(data):
        return create_leaf([record[0] for record in leftsplit])
    if len(rightsplit)==len(data):
        return create_leaf([record[0] for record in rightsplit])

    #recursive calls to decision_tree_create()
    left_tree = decision_tree_create(leftsplit,max_depth,current_depth + 1)
    right_tree = decision_tree_create(rightsplit,max_depth,current_depth + 1)
    return {'is_leaf':False,'prediction':None,'splitting_feature':split_feature[0],'split_feature_value':split_feature[1],'left':left_tree,'right':right_tree}



#function to classify a single data
def classify(tree, x, annotate = False):
    if tree['is_leaf']:
        if annotate:
            # predicting the majority class in the leaf
            return tree['prediction']

    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
             # print("Split on %s = %s" % (tree['splitting_feature'], split_feature_value))
            pass
        if(x[tree['splitting_feature']] == tree['split_feature_value']):
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'],x,annotate)


#function to predict the data based upon the supplied tree
def evaluate_tree(tree,data):
    prediction = []
    for record in data:
        prediction.append(classify(tree,record,annotate=True))
    return prediction

#function to test accuracy of the model by comparing with the true label of test data
def accuracy(prediction,test_data):
    correct = 0
    cl=[]
    tp=[]
    tn=[]
    fp=[]
    fn=[]
    true = 0
    false = 0
    j=0
       #calculating accuracy by comparing predicted value and actual value
    for i in test_data:
        if i[0]==prediction[j]:
            cl.append(1)
            if prediction[j] == '1':
                tp.append(1)
            else:
                tn.append(1)
            j+=1
        else:
            cl.append(0)
            if prediction[j]=='0':
                fn.append(1)
            else:
                fp.append(1)
            j+=1

    for k in range(0,len(cl)-1):
        if cl[k]==1:
            true+=1
        else:
            false+=1
    Accuracy=(true)/(true+false)
    Accuracy1=Accuracy*100
    Error=1-Accuracy
    Error1=Error*100
    print("-----------------------Confusion Matrix-----------------------------")

    print("True Positive:",len(tp))
    print("True Negative:",len(tn))
    print("False Positive:",len(fp))
    print("False Negative:",len(fn))

    return Accuracy1


#function to create bags out the train data
def create_bag(data,number_of_bags):
    bag_list=[]
    bag=[]
    for i in range(0,number_of_bags):
        bag=[]
        for j in range(0,int(len(data)/number_of_bags)):
            bag.append(data[random.randint(0,len(data)-1)])

        bag_list.append(bag) #adding the bags to a list of list
    return bag_list

#function to learn a tree for each bag
def create_tree_for_each_bag(bag_list,max_depth):
    tree_list=[]
    # tree =[]
    for i in range(0,len(bag_list)):
        tree=[]
        tree = decision_tree_create(bag_list[i],max_depth)   #vary the depth to build tree of different depths...
        tree_list.append(tree)
    return  tree_list


#function to get prediction from all the trees learnes
def evaluate_multiple_trees(tree_list,test_data):
    prediction_list=[]
    for i in range(0,len(tree_list)):
        prediction = []
        prediction = evaluate_tree(tree_list[i],test_data)
        prediction_list.append(prediction) #add the prediction from each tree to a list of list
    return prediction_list

#function to get the final prediction by majority voting
def majority_prediction(prediction_list):
    t=[]
    for col in range(0,len(prediction_list[0])):
        column=[]
        for temp in prediction_list:
            c=temp[col]
            column.append(c)
        try:
            tp=mode(column)
        except:
            tp='1'
        t.append(tp)
    return t

