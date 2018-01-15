__author__ = 'yatinsharma'
import nltk

#function to open text file
def file_open(path):
    f = open(path,'rU')
    text = f.read().split()
    return text

#function to return most common words
def frequency_distribution(text,n):

    fdist = nltk.FreqDist(text)
    # print(type(fdist))
    # fdist.plot(n, cumulative=True)               #plots the distribution of 50 most common words

    return fdist.most_common(n)



def create_suffix(text,n=10):
    suffix_freq_dist = nltk.FreqDist()
    for word in text:
        word.lower()
        suffix_freq_dist[word[-1:]] +=1
        suffix_freq_dist[word[-2:]] +=1
        suffix_freq_dist[word[-3:]] +=1
    print(suffix_freq_dist.most_common(n))
    common_suffix = [suffix for (suffix,count) in suffix_freq_dist.most_common(n)]
    return common_suffix

def create_pos_features(word,common_suffix):
    features = {}
    for suffix in common_suffix:
        features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
        return features

def pos_text_features(text,gender,common_suffix):
    # token = text.lower().split()                                     #tokenise words into a list
    token = [t for t in text if("#" in t or t.isalnum())]           #remove all words except #tags and alpaha-numeric
    # print("yatin---------")
    # print(token)
    features = {'Gender':gender}
    for suffix in common_suffix:
        # features['contains({})'.format(word)] = (word in token)
        for x in token:

            if x.lower().endswith(suffix[0]):
                features['endswith({})'.format(suffix[0])] = True
                print(x)
                break
            # elif not x.lower().endswith(suffix[0]):
            #     features['endswith({})'.format(suffix[0])] = False
            else:
                features['endswith({})'.format(suffix[0])] = False

    # print(features)
    return features


text = file_open('/Users/yatinsharma/PycharmProjects/Text Mininng/all_words.txt')
# most_common_words = frequency_dis
common_suffix = create_suffix(text,n=3)
features = pos_text_features(['he','hene','tries'],'male',common_suffix)
print(features)
