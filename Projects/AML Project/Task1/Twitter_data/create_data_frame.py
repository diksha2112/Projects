__author__ = 'yatinsharma'
__author__ = 'yatinsharma'
import os
import nltk
import xml.etree.cElementTree as ET
import re
import random
import pandas as pd
import preprocessor as p
import Get_file_paths
import Common_words
from unidecode import unidecode


all_words = []
labeled_text = []
all_words_male=[]
all_words_female = []

def create_corpus(files):

    for file in files:
        try:
            tree = ET.parse(file)
            root = tree.getroot()
        except:
            continue
        for i,document in enumerate(root.iter('document')):
            text = document.text
            try:
                text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
                # text = re.sub('><(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '$', text)
            except:
                continue
            token = text.lower().split()                                                 #tokenise words into a list
            token = [t for t in token if("#" in t or t.isalnum())]           #remove all words except #tags and alpaha-numeric
            # for word in token:
            #     if word not in all_words:
            #         all_words.append(word)
        # print(all_words)

def create_labeled_text(files,label):
    for file in files:
        # print("test--------------")
        # print(file)
        try:
            tree = ET.parse(file)
            root = tree.getroot()
        except:
            continue
        for i,document in enumerate(root.iter('document')):
            text = document.text
            try:
                text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
                # text = re.sub('><(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '$', text)
            except:
                continue
            token = text.lower().split()                                                 #tokenise words into a list
            token = [t for t in token if("#" in t or t.isalnum())]           #remove all words except #tags and alpaha-numeric
            labeled_text.append((token,label))
            for word in token:
                if word not in all_words:
                    all_words.append(word)

    random.shuffle(labeled_text)

def create_labeled_text_userwise(files,label):
    for i,file in enumerate(files):
        print("creating labeled text for %s user:%s"%(label,i))
        labeled_text2 =[]
        # print("test--------------",label)
        # print(file)
        try:
            tree = ET.parse(file)
            root = tree.getroot()
        except:
            continue
        for i,document in enumerate(root.iter('document')):
            text = document.text
            try:
                text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
                # text = re.sub('><(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '$', text)
            except:
                continue

            token = text.lower().split()                                                 #tokenise words into a list
            token = [t for t in token if("#" in t or t.isalnum())]           #remove all words except #tags and alpaha-numeric
            for word in token:
                if("#" in word or word.isalnum()):
                    labeled_text2.append(word)
                    all_words_male.append(word)
        labeled_text.append((labeled_text2,label))

    random.shuffle(labeled_text)
    # print(labeled_text)

def common_words(n):
    text = Common_words.file_open('/Users/yatinsharma/PycharmProjects/Text Mininng/all_words.txt')
    common_word= Common_words.frequency_distribution_words(text,n)
    return common_word

def common_suffixes(n):
    text = Common_words.file_open('/Users/yatinsharma/PycharmProjects/Text Mininng/all_words.txt')
    common_suffixes = Common_words.frequency_distribution_suffixes(text,n)
    return common_suffixes

def average_word_length(word_list):
    total_length = sum(len(word) for word in word_list)
    num_words = len(word_list)
    Average = total_length/num_words
    if Average <=4.5:
        Avg_category = 'Low'
    elif Average >4.5 and Average<6.0:
        Avg_category = 'Medium'
    elif Average >6:
        Avg_category = 'high'
    else:
        Avg_category = 'Unknown'

    return Avg_category

def distinct_words(word_list):
    Distinct_words = len(set(word_list))

    if Distinct_words <=1000:
        category = 'Low'
    elif Distinct_words >1000 and Distinct_words<1500:
        category = 'Medium'
    elif Distinct_words >1500:
        category = 'high'
    else:
        category = 'Unknown'


    return category



def text_features(text,gender,common_word):
    # token = text.lower().split()                                     #tokenise words into a list
    token = [t for t in text if("#" in t or t.isalnum())]           #remove all words except #tags and alpaha-numeric
    features = {'Gender':gender}
    # average_word_length = average_word_length(token)
    features['Avg_word_length']=average_word_length(token)
    features['Distinct_words'] = distinct_words(set(token))
    for word in common_word:
        # features['contains({})'.format(word)] = (word in token)
        if word[0] in token:
            features['contains({})'.format(word[0])] = True
        else:
            features['contains({})'.format(word[0])] = False
    return features

def POS_features(text,gender,common_suffixes):
    # token = text.lower().split()                                     #tokenise words into a list
    token = [t for t in text if("#" in t or t.isalnum())]           #remove all words except #tags and alpaha-numeric
    features = {'Gender':gender}
    for suffix in common_suffixes:
        # features['contains({})'.format(word)] = (word in token)

        for x in token:

            if x.lower().endswith(suffix):
                features['endswith({})'.format(suffix)] = True
                break

            else:
                features['endswith({})'.format(suffix)] = False
    return features




def feature_set_words(document,no_of_words=10):
    text = Common_words.file_open('/Users/yatinsharma/PycharmProjects/Text Mininng/all_words.txt')
    common_word= Common_words.frequency_distribution_words(text,no_of_words)
    print(common_words)
    word_feature_set = [(text_features(a,b,common_word)) for (a,b) in document if len(a) >0]
    return  word_feature_set

def feature_set_POS(document,no_of_POS=10):
    text = Common_words.file_open('/Users/yatinsharma/PycharmProjects/Text Mininng/all_words.txt')
    common_suffixes = Common_words.frequency_distribution_suffixes(text,no_of_POS)
    print(common_suffixes)
    POS_feature_set = [(POS_features(a,b,common_suffixes)) for (a,b) in document if len(a) >0]
    return POS_feature_set


def create_dataFrame_csv(feature_sets,filename):
    df=pd.DataFrame(feature_sets)
    df.to_csv(filename+'.csv')


def create_test_train(set,train_portion):
    print("-------------------Creating random train and test sample-------------------------")
    import random as rnd
    import math
    train_length = math.floor(train_portion*len(set))
    rnd.shuffle(set)
    train_set = set[:train_length]
    test_set = set[train_length:]

    return train_set,test_set





# male_files,female_files = get_files('/Users/yatinsharma/PycharmProjects/Text Mininng/Gender Classification')
male_files,female_files = Get_file_paths.get_files('/Users/yatinsharma/PycharmProjects/Text Mininng/All_XMLS')
create_labeled_text_userwise(male_files,'Male')
create_labeled_text_userwise(female_files,'Female')



feature_sets_words = feature_set_words(labeled_text,no_of_words=100)
# feature_sets_POS = feature_set_POS(labeled_text,no_of_POS= 10)
# create_dataFrame_csv(feature_sets_POS,'twitter_50_POS')
create_dataFrame_csv(feature_sets_words,'twitter_categorical_avg_length_distinct')



