__author__ = 'yatinsharma'
__author__ = 'yatinsharma'

import math
import pandas as pd
import numpy as np

#function to read file
def read_data(floc):
    import csv

    with open(floc, 'r') as f:
        reader = csv.reader(f)
        list_data = list(reader)
        list_data.__delitem__(0)
        return list_data


#function to initialize weight of 1/n to each data point, creating extra column in the end of the data to store the weighs
def initialize_weight(data):
    n=len(data)
    for i in data:
        i.append(1/n)
    return data



def uniquecounts(data):
    results = {}
    for row in data:
        # The result is the last column
        r = row[0]
        if r not in results:
            results[r] = 0
        else:
            results[r] += 1
    return results

#function to calculate entropy based upon the weights(last column) of each data point
def entropy(data):
    val_freq = {}
    data_entropy = 0.0
    subset_entropy = 0.0
    for record in data:
        #suming the weights for each label in data
        if (record[0] in val_freq):
            val_freq[record[0]] += record[len(record)-1]

        else:
            val_freq[record[0]] = record[len(record)-1]

    for freq in val_freq.values():
        # calculating the entropy based upon the weights of the class labels
        try:
            data_entropy += (-freq /sum(val_freq.values()) ) * math.log(freq / sum(val_freq.values()), 2)
        except:
            data_entropy=0

    return data_entropy




#function to get the best feature based upon the info gain
def best_feature(rows):  #rows is the set, either whole dataset or part of it in the recursive call,
    #scoref is the method to measure heterogeneity. By default it's entropy.
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
    for col in range(1, column_count-1):
        val_freq = {}
        for record in t:
            if (record[col] in val_freq):
                val_freq[record[col]] += 1.0
            else:
                val_freq[record[col]] = 1.0


        for val in val_freq:

            weight = val_freq[val] / sum(val_freq.values())
            subset1 = [record for record in t if record[col] == val]
            subset2 = [record for record in t if record[col] != val]
            subset_entropy = weight * entropy(subset1) + (1 - weight) * entropy(subset2)
            info_gain = orig_entropy - subset_entropy
            #choosing the best feature based upon the highest info gain
            if info_gain > best_gain and len(subset1) > 0 and len(subset2) > 0:
                best_gain = info_gain
                best_attribute = (col, val)
                best_subset1 = subset1
                best_subset2 = subset2

            else:
                pass

    return(best_attribute,best_subset1,best_subset2)


#function to create a leaf node
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


#function to create decision tree of desired depth using recursion
def decision_tree_create(data,max_depth = 1,current_depth = 0):
    leftsplit = []
    rightsplit=[]

    if entropy(data)==0:
        return create_leaf([record[0] for record in data])


    if current_depth >= max_depth:
        return create_leaf([record[0] for record in data])
#finding the best feature by calling best_feature() function
    split_feature,leftsplit,rightsplit =best_feature(data)


    if(len(leftsplit)==0 and len(rightsplit)==0):
        return create_leaf([record[0] for record in data])
    if(len(leftsplit))==0:
        # creating leaf node,if left data is empty
        return create_leaf([record[0] for record in rightsplit])
    if(len(rightsplit)==0):
        # creating leaf node,if right data is empty
        return create_leaf([record[0] for record in leftsplit])


#create a leaf node ,if split is perfect
    if len(leftsplit)==len(data):
        return create_leaf([record[0] for record in leftsplit])
    if len(rightsplit)==len(data):
        return create_leaf([record[0] for record in rightsplit])

#recursive calls to decision_tree_create function
    left_tree = decision_tree_create(leftsplit,max_depth,current_depth + 1 )
    right_tree = decision_tree_create(rightsplit,max_depth,current_depth + 1)
    return {'is_leaf':False,'prediction':None,'splitting_feature':split_feature[0],'split_feature_value':split_feature[1],'left':left_tree,'right':right_tree}



#function to classify a single  data point based upon a tree
def classify(tree, x):
    if tree['is_leaf']:
            # predicting the majority class at leaf
            return tree['prediction']

    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
             # recursive call based upon the feature value at current node
        if(x[tree['splitting_feature']] == tree['split_feature_value']):
            return classify(tree['left'], x)
        else:
            return classify(tree['right'],x)


#function to classify the entire data based upon a tree,
def evaluate_tree(tree,data):
    prediction = []
    #sending each data for classification to above classify() function
    for record in data:
        prediction.append(classify(tree,record))
    return prediction

#function to test accuracy of our prediction by comparing with actual class labels of test data
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

#function to calculate the weighted error of the classifier
def weighted_Error(prediction,train_data):
    correct = 0
    cl=[]
    tp=[]
    tn=[]
    fp=[]
    fn=[]
    true = 0
    false = 0
    j=0
    # print(correct/len(prediction))
    weight_mistake_sum = 0
    weight_all_sum = 0
       #calculating weighted error of the classifier
    for i in train_data:
        weight_all_sum += i[len(i)-1]
        if i[0]==prediction[j]:
            cl.append(1)
            if prediction[j] == '1':
                tp.append(1)
            else:
                tn.append(1)
            j+=1
        else:
            cl.append(0)
            weight_mistake_sum += i[len(i)-1]
            if prediction[j]=='0':
                fn.append(1)
            else:
                fp.append(1)
            j+=1
    weighted_error = float(weight_mistake_sum)/float(weight_all_sum)
    return weighted_error,weight_all_sum,cl

#function to create adaboost trees
def adaboost_tree_create(data,number_of_trees,max_depth):
    tree_list=[]
    weights =[]
    for t in range(number_of_trees):
        if(entropy(data)==0):
            weights.append(weight)
            tree_list.append(tree)
        else:
            tree = decision_tree_create(data,max_depth)
            tree_list.append(tree)
            prediction= evaluate_tree(tree,data)
            weight_error = weighted_Error(prediction,data)[0]
            weight_all_sum = weighted_Error(prediction,data)[1]
            classification = weighted_Error(prediction,data)[2]
            try:
                weight = (1/2)* math.log(((1-weight_error)/weight_error),math.e)
            except:
                weight =0.01
            weights.append(weight)
            x=0
            for j in classification:
                if j ==1:
                    data[x][len(data[x])-1]=(data[x][len(data[x])-1])* math.exp(weight)
                else:
                    data[x][len(data[x])-1]=(data[x][len(data[x])-1])* math.exp(-weight)
                x+=1

            sum_weights =0.0

            for i in data:
                sum_weights +=i[len(i)-1]
                i[len(i)-1]=(i[len(i)-1])/sum_weights

    return tree_list,weights


#function to predict based upon the trees learned by the adaboost algorithm
def predict_adaboost(tree_list,weights,data):
    prediction_list=[]
    for i,tree in enumerate(zip(tree_list,weights)):
        prediction=[]

        prediction = evaluate_tree(tree[0],data)
        prediction[:] = [int(x) for x in prediction]
        prediction[:] = [x*tree[1] for x in prediction]
        prediction_list.append(prediction)

    return prediction_list

#funtion to get the final prediction by summing the prediction of each tree and testing if greater than or less than zero
def weighted_prediction(prediction_list):
    t=[]
    for col in range(0,len(prediction_list[0])):
        column=[]
        for temp in prediction_list:
            c=temp[col]
            column.append(c)

            tp=sum(column)

            if tp>=0:
                t.append('1')
            else:
                t.append('-1')
    return t











