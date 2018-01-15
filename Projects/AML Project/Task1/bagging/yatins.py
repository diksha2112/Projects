__author__ = 'yatinsharma'
import math
import pandas as pd
import numpy as np
import os
import sys
import decision_Tree_Boosting_run
import Decision_tree_bagging_final

def read_data(filepaths):
    data_list=[]
    for f in filepaths:
        import csv
        if f.endswith(".csv"):
            with open(f, 'r') as f:
                reader = csv.reader(f)
                list_data = list(reader)
                list_data.__delitem__(0)
                # print(list_data)
                data_list.append(list_data)
    return data_list


def get_files(directory):

    file_paths = []  # List for storing all the file path in the given directory.

        # traversing the directory.
    for root, directories, files in os.walk(directory):
        for filename in files:
        # Joining  the two strings to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Adding to the list

    data_train_test = read_data(file_paths)
    return data_train_test  # returning list of train and test data



if __name__ == "__main__":
    #getting test and train data in a list of list,where[0]=test,[1]=train

    test_train_list = get_files(sys.argv[4]) #getting the test and train data in the list
    #evaluating boosting algorithm
    if(sys.argv[1]=='boost'):
        print("Evaluating boosting algorithm at depth: %s and number of trees = %s ........Please wait....." % (sys.argv[2],sys.argv[3]))
        train_data_weighted = decision_Tree_Boosting_run.initialize_weight(test_train_list[1])   #intializing weight of 1/n to each data point
        test_data = test_train_list[0] #loading the test data
        tree_list,weights=decision_Tree_Boosting_run.adaboost_tree_create(train_data_weighted,number_of_trees=int(sys.argv[3]),max_depth=int(sys.argv[2]))#creating the desired number of boosting ensemble
        prediction_list=decision_Tree_Boosting_run.predict_adaboost(tree_list,weights,test_data) #predicting the test labels using each tree
        prediction=decision_Tree_Boosting_run.weighted_prediction(prediction_list) # getting the final prediction by giving weight to each tree learnt
        accuracy=decision_Tree_Boosting_run.accuracy(prediction,test_data) #testin the accuracy of the final boosting model
        print("---------------------------------------------------------------------")
        print("Final Accuracy:",accuracy,"%")
        print("---------------------------------------------------------------------")
    #evaluating bagging algorithm
    if(sys.argv[1]=='bag'):
        print("Evaluating bagging algorithm at depth: %s and number of bags = %s ........Please wait....." % (sys.argv[2],sys.argv[3]))
        train_data = test_train_list[1]
        test_data = test_train_list[0]
        bag_list = Decision_tree_bagging_final.create_bag(train_data,number_of_bags=int(sys.argv[3]))   #creating bags
        tree_list=Decision_tree_bagging_final.create_tree_for_each_bag(bag_list,max_depth=int(sys.argv[2])) #creating trees for each bag
        prediction_list=Decision_tree_bagging_final.evaluate_multiple_trees(tree_list,test_data) #evaluating each tree and storing the prediction of each tree in a list of list
        final_prediction = Decision_tree_bagging_final.majority_prediction(prediction_list) #getting the final prediction by majority vote
        Accuracy=Decision_tree_bagging_final.accuracy(final_prediction,test_data)  #testing the accuracy the final model
        print("---------------------------------------------------------------------")
        print("Final Accuracy:",Accuracy,"%")
        print("---------------------------------------------------------------------")






