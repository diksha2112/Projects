__author__ = 'yatinsharma'

import os
import nltk
import xml.etree.cElementTree as ET
import re

from unidecode import unidecode

all_words = []

def create_corpus(files):

    for file in files:
        # print("test--------------")
        # print(file)
        tree = ET.parse(file)
        root = tree.getroot()
        for i,document in enumerate(root.iter('document')):
            text = document.text
            try:
                text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
            except:
                continue
            token = text.lower().split()                                                 #tokenise words into a list
            token = [t for t in token if("#" in t or t.isalnum())]           #remove all words except #tags and alpaha-numeric
            for word in token:
                all_words.append(word)

    print(len(all_words))
    # print(all_words[0:10])
    f = open('all_words.txt','w')
    string = " ".join(all_words)
    f.write(unidecode(string))

def get_files(directory):

    for f in os.listdir(directory):
        if f.endswith('.xml'):
            file_list.append(os.path.join(directory,f))
    print(len(file_list))

    return file_list


file_list = get_files('/Users/yatinsharma/PycharmProjects/Text Mininng/Corpus/pan16-author-profiling-training-dataset-english-2016-04-25')
create_corpus(file_list)

