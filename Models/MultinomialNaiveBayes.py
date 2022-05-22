from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


class MultinomialNaiveBayes:

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

            self.__vectorizer = TfidfVectorizer(min_df=5,max_df=0.8,sublinear_tf=True,use_idf=True,stop_words=stopwords)

            train_vectors = self.__vectorizer.fit_transform([sample[0] for sample in self.__lemmatizeTrainingData])

            self.__model = MultinomialNB()
            self.__model.fit(train_vectors, [sen[1] for sen in self.__lemmatizeTrainingData])


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