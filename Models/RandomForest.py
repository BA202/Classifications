from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

class RandomForest:

    def __init__(self,trainingData= None,**kwargs):
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

        if not self.__trainingData == None:
            self.__lemmatizeTrainingData = []

            for data in self.__trainingData:
                self.__lemmatizeTrainingData.append([self.cleanUp(data[0]),data[1]])

            self.__vectorizer = TfidfVectorizer(max_features=2500,min_df=5,max_df=0.8,sublinear_tf=True,use_idf=True,stop_words=stopwords)

            train_vectors = self.__vectorizer.fit_transform([sample[0] for sample in self.__lemmatizeTrainingData])

            gridSearchParmeters = {'max_features': ('auto','sqrt'),
             'n_estimators': [500, 1000, 1500],
             'max_depth': [5, 10, None],
             'min_samples_split': [5, 10, 15],
             'min_samples_leaf': [1, 2, 5, 10]}

            grid_search = GridSearchCV(RandomForestClassifier(), gridSearchParmeters,
                                       cv=5, return_train_score=True,
                                       n_jobs=-1)
            grid_search.fit(train_vectors,[sample[1] for sample in self.__lemmatizeTrainingData])

            print("best param are {}".format(grid_search.best_params_))
            means = grid_search.cv_results_['mean_test_score']
            stds = grid_search.cv_results_['std_test_score']
            for mean, std, param in zip(means, stds,grid_search.cv_results_['params']):
                print("{} (+/-) {} for {}".format(round(mean, 3), round(std, 2),param))

            self.__model = RandomForestClassifier(
                max_features=grid_search.best_params_['max_features'],
                max_depth=grid_search.best_params_['max_depth'],
                n_estimators=grid_search.best_params_['n_estimators'],
                min_samples_split=grid_search.best_params_[
                    'min_samples_split'],
                min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                bootstrap=grid_search.best_params_['bootstrap'])
            self.__model.fit(train_vectors, [sample[1] for sample in self.__lemmatizeTrainingData])


    def classify(self,sentence):
        vec = self.__vectorizer.transform([self.cleanUp(sentence)])
        prediction = self.__model.predict(vec)
        return prediction[0]


    def cleanUp(self,sen):
        sen = sen.lower()
        lem = self.__stemmer.stem(sen)
        return lem

    def getParameters(self):
        return self.__model.get_params()