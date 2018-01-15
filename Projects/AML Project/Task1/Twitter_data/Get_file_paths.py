__author__ = 'yatinsharma'

import os

#function to check actual gender in truth table
def lookup_truth(lookup):
    file = open('/Users/yatinsharma/PycharmProjects/Text Mininng/pan16-author-profiling-training-dataset-english-2016-04-25/truth.txt','r')
    gender = str
    print("searching for %s" %lookup)
    for line in file:
        if lookup in line:
            print("found")
            print(line[35])
            gender = line[35]
            break
    file.close()

    return gender

#function to get file path of males and females tweets according to the truth table
def get_files(path):
    male_file_paths = []
    female_file_paths = []
    all_file_paths = []

    print(os.listdir(path))

    for f in os.listdir(path):
        print(f[:-4])
        if f.endswith('.xml'):
            gender = lookup_truth(f[:-4])

            if gender == 'M':
                male_file = os.path.join(path,f)
                male_file_paths.append(male_file)

            if gender == 'F':
                female_file = os.path.join(path,f)
                female_file_paths.append(female_file)


    return male_file_paths,female_file_paths

# male_file_paths,female_file_paths = get_files('/Users/yatinsharma/PycharmProjects/Text Mininng/pan16-author-profiling-training-dataset-english-2016-04-25')
# print(len(male_file_paths))
# print(len(female_file_paths))
#
# print(male_file_paths)
# print(female_file_paths)