import nltk
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from numpy import argmax
from numpy import array
from numpy import argmax
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier


class Baseline2(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.model=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)


        # self.model=RandomForestClassifier(n_estimators= 100, max_depth=9, min_samples_split=110,
        #                           min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10)
        # self.model=GradientBoostingClassifier(learning_rate=0.12, n_estimators=80,max_depth=7, min_samples_leaf =60, 
        #        min_samples_split =1200, max_features='sqrt', subsample=0.8, random_state=10)
        # self.model = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
        #     max_features=20, max_leaf_nodes=None,
        #     min_impurity_split=0.005, min_samples_leaf=3,
        #     min_samples_split=2, min_weight_fraction_leaf=0.0,
        #     presort=False, random_state=1, splitter='random'
        #     )
        # self.model=AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
        #                  algorithm="SAMME",
        #                  n_estimators=200, learning_rate=0.1)

    def extract_features(self, word):
        global pos_tag
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        len_set=len(set(word))**0.3
        len_synonym=len(wn.synsets(word))
        #word=nltk.word_tokenize(word)
        a=pos_tag([word])
        list02=[]
        for i in a:
            list02.append(i[1])
        data = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ','JJR',',','JJS','(', 'MD', 'NN','NNS','NNP','NNPS','PDT','POS',"''",'.',
                'PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB',':',
                'LS','$']
        values = array(data)
        # integer encode
        label_encoder = LabelEncoder()
        data_encoded = label_encoder.fit_transform(values)
        
        dictionary = dict(zip(data, data_encoded))
        #list03=[]
        #for i in list02:
            # list03.append(dictionary[i])
        pos_num=dictionary[a[0][1]]
        
        return [len_chars, len_tokens,pos_num,len_set,len_synonym]

    def extract_features_tf(self,trainset,testset):
        list01=[]
        for sent in trainset+testset:
            #list01.append("".join(sent['sentence']))
            list01.append(sent['sentence'].split())
        list02=[]
        for i in list01:
            for j in i:
                list02.append(j)
            
        freq_dict=nltk.FreqDist(list02)
        return freq_dict



    def train(self, trainset,testset):
        X = []
        y = []
        dict1=self.extract_features_tf(trainset,testset)
        for sent in trainset+testset:
            temp=self.extract_features(sent['target_word'])*5
            temp.append(dict1[sent['target_word']])
            X.append(temp)
            #X.append([dict1[sent['target_word']]])
            y.append(sent['gold_label'])
        
        self.model.fit(X, y)

    def test(self,trainset,testset):
        X = []
        dict2=self.extract_features_tf(trainset,testset)
        for sent in testset:
            temp2=self.extract_features(sent['target_word'])*5
            temp2.append(dict2[sent['target_word']])
            X.append(temp2)
            #X.append([dict2[sent['target_word']]])
        return self.model.predict(X)




class Baseline1(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.model = svm.RandomForestClassifier()

    def extract_features(self, word):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        pos_tag=pos_tag(word)
        
        return [len_chars, len_tokens,pos_tag]

    def extract_features_tf(self,trainset):
        list01=[]
        for sent in trainset:
            #list01.append("".join(sent['sentence']))
            list01.append(sent['sentence'].split())
        list02=[]
        for i in list01:
            for j in i:
                list02.append(j)
            
        freq_dict=nltk.FreqDist(list02)
        return freq_dict
        
    def train(self, trainset):
        X = []
        y = []
        dict1=self.extract_features_tf(trainset)
        for sent in trainset:
            X.append([dict1[sent['target_word']]])
            y.append(sent['gold_label'])
            
        self.model.fit(X, y)

    def test(self, testset):
        X = []
        dict2=self.extract_features_tf(testset)
        for sent in testset:
            X.append([dict2[sent['target_word']]])

        return self.model.predict(X)







class Ann(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.model = MLPClassifier()

    def extract_features(self, word):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))

        return [len_chars, len_tokens]

    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:
            y.append(sent['gold_label'])
            X.append(sent.drop('gold_label'),axis=1)

        self.model.fit(X, y)

    def test(self, testset):
        X = []
        for sent in testset:
            X.append(sent.drop('gold_label'),axis=1)
    

        return self.model.predict(X)


      