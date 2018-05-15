import sklearn


def report_score(gold_labels, predicted_labels, detailed=False):
    macro_F1 = sklearn.metrics.f1_score(gold_labels, predicted_labels, average='macro')
    print("macro-F1: {:.2f}".format(macro_F1))
    if detailed:
        scores = sklearn.metrics.precision_recall_fscore_support(gold_labels, predicted_labels)
        print("{:^10}{:^10}{:^10}{:^10}{:^10}".format("Label", "Precision", "Recall", "F1", "Support"))
        print('-' * 50)
        print("{:^10}{:^10.2f}{:^10.2f}{:^10.2f}{:^10}".format(0, scores[0][0], scores[1][0], scores[2][0], scores[3][0]))
        print("{:^10}{:^10.2f}{:^10.2f}{:^10.2f}{:^10}".format(1, scores[0][1], scores[1][1], scores[2][1], scores[3][1]))
    print()





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



