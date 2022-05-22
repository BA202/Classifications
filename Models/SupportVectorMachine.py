from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from nltk.stem import PorterStemmer
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import fbeta_score, make_scorer


class SupportVectorMachine:

    def __init__(self,trainingData= None, kernel = 'linear',debug = False,**kwargs):
        stopwords = ['from', 'him', 'shouldn', 'will', 'than', 'what', 'weren',
                     'over', "shouldn't", 'in', 'its', 'above', 'o', 'about',
                     'has', 'or', 'off', 'before', 'doesn', "you've", 'just',
                     'but', 'my', 'd', 'having', 'they', "aren't", 'ourselves',
                     'ain', 'with', "haven't", 'it', 'under', 'after',
                     'myself', 'did', "you'd", "won't", 'which', 'theirs',
                     've', 'ours', 'haven', "mustn't", 'same', "it's",
                     'mightn', 'for', 'our', 'how', "don't", 'both', 'them',
                     'those', 'ma', 'she', 'any', 'once', 'couldn', 'these',
                     'itself', 'then', "doesn't", 'i', 'here', 'shan', 'so',
                     'hadn', 'who', 'into', "she's", 'yourselves', 'me', 'all',
                     'mustn', 'own', 'isn', 'needn', 'y', 'be', "weren't",
                     'against', 'up', 'whom', 'where', 'wouldn', 'by',
                     "hasn't", "you'll", 'because', 'at', 'such', "you're",
                     'their', 'down', 'doing', 'through', 'not', "shan't",
                     'is', "couldn't", 'most', 'to', "needn't", 'again', 's',
                     'was', 'until', 'no', 'themselves', 'and', 'few',
                     "mightn't", 'does', 'out', 'too', "that'll", 'between',
                     're', 'further', 'why', 'aren', 'his', 'll', 'himself',
                     'while', 'should', "should've", 'there', 'nor', 'he',
                     'yourself', 'other', "wasn't", 'more', 'her', 'very',
                     'have', 'during', 'only', 'hasn', 'when', 'an', 'didn',
                     'below', 'on', 'being', "isn't", 'had', 'that', 'do',
                     'of', 'the', 'were', 'now', "wouldn't", 'your', 'some',
                     't', 'we', 'won', 'yours', "didn't", 'been', 'm', 'as',
                     'wasn', 'if', "hadn't", 'am', 'are', 'this', 'you',
                     'hers', 'each', 'don', 'can', 'a', 'herself']

        self.__trainingData = trainingData
        self.__stemmer = PorterStemmer()
        self.__doGridSearch = False

        if 'param' in kwargs.keys():
            kernel = kwargs['param']

        if kernel == "rbf":
            self.__doGridSearch = True

        elif kernel == "poly":
            self.__doGridSearch = True

        elif kernel == "sigmoid":
            self.__doGridSearch = True


        if not self.__trainingData == None:
            self.__vectorizer = TfidfVectorizer(max_features=2500,min_df=5,max_df=0.8,sublinear_tf=True,use_idf=True,stop_words=stopwords)

            train_vectors = self.__vectorizer.fit_transform([sample[0] for sample in self.__trainingData])
            if self.__doGridSearch:
                score = make_scorer(self.lossFunction,
                                    greater_is_better=False)
                if kernel == "rbf" or kernel == "poly" or kernel == "sigmoid":
                    C_range = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
                    gamma_range = [1,1.2,1.3,1.4,1.5,1.6,1.7]
                    degree = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                    gridSearchParmeters = dict(gamma=gamma_range, C=C_range,kernel = [kernel],degree= degree,class_weight=['balanced'])
                else:
                    gridSearchParmeters = {}
                    raise ValueError("Not able to perform grid search!")

                grid_search = GridSearchCV(svm.SVC(),
                                           gridSearchParmeters,
                                           cv=10, return_train_score=True,
                                           n_jobs=-1)
                grid_search.fit(train_vectors, [sample[1] for sample in
                                                self.__trainingData])

                print("best param are {}".format(grid_search.best_params_))
                means = grid_search.cv_results_['mean_test_score']
                stds = grid_search.cv_results_['std_test_score']
                for mean, std, param in zip(means, stds,
                                            grid_search.cv_results_['params']):
                    print("{} (+/-) {} for {}".format(round(mean, 3),
                                                      round(std, 2), param))

                self.__model = svm.SVC(gamma=grid_search.best_params_['gamma'],
                C=grid_search.best_params_['C'],
                kernel=grid_search.best_params_['kernel'],
                                       degree=grid_search.best_params_['degree'],class_weight='balanced')
                self.__model.fit(train_vectors, [sample[1] for sample in
                                                 self.__trainingData])

            else:
                self.__model = svm.SVC(kernel=kernel)
                if debug:
                    print(f"---SupportVectorMachine with kernel: {kernel}")
                self.__model.fit(train_vectors, [sample[1] for sample in self.__trainingData])


    def classify(self,sentence):
        senVector = self.__vectorizer.transform([self.cleanUp(sentence)])
        prediction = self.__model.predict(senVector)
        return prediction[0]



    def cleanUp(self,sen):
        sen = sen.lower()
        lem = self.__stemmer.stem(sen)
        return lem

    @staticmethod
    def lossFunction(y_true, y_pred):
        error = 0
        for yT,yP in zip(y_true,y_pred):
            factor = 1
            if yT == "Negative":
                factor = 100

            if not yT == yP:
                error += factor
        return error

    def getParameters(self):
        modelParams = self.__model.get_params()
        return {'kernel':modelParams['kernel'],'degree':modelParams['degree'],'gamma':modelParams['gamma'],'C':modelParams['C'],'max_iter':modelParams['max_iter']}
