__author__ = 'yatinsharma'
import nltk

#function to open text file
def file_open(path):
    f = open(path,'rU')
    text = f.read().split()
    return text

#function to return most common words
def frequency_distribution_words(text,n):
    a = []
    for word in text:
        if len(word)>=6:
            a.append(word)
    fdist = nltk.FreqDist(a)
    # print(type(fdist))
    # fdist.plot(n, cumulative=True)               #plots the distribution of 50 most common words
    return fdist.most_common(n)

def frequency_distribution_suffixes(text,n):
    suffix_freq_dist = nltk.FreqDist()
    for word in text:
        word.lower()
        suffix_freq_dist[word[-1:]] +=1
        suffix_freq_dist[word[-2:]] +=1
        suffix_freq_dist[word[-3:]] +=1
    # print(suffix_freq_dist.most_common(n))
    common_suffix = [suffix for (suffix,count) in suffix_freq_dist.most_common(n)]
    return common_suffix



text_Male = file_open('/Users/yatinsharma/PycharmProjects/Text Mininng/all_words_Male.txt')
text_Female = file_open('/Users/yatinsharma/PycharmProjects/Text Mininng/all_words_Female.txt')

most_common_words_Male = frequency_distribution_words(text_Male,20)
most_common_words_Female = frequency_distribution_words(text_Female,20)

print(most_common_words_Male)
print(most_common_words_Female)
# most_common_suffixes = frequency_distribution_suffixes(text,100)