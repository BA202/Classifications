from sklearn.naive_bayes import MultinomialNB
from StatisticalMLVectorizer import StatisticalMLVectorizer


class MultinomialNaiveBayes:

    def __init__(self,trainingData= None,**kwargs):
        self.__vectorizer = StatisticalMLVectorizer()
        self.__alpha = 1

        features = self.__vectorizer.vectorize([sample[0] for sample in trainingData])
        labels = [sample[1] for sample in trainingData]

        self.__model = MultinomialNB(alpha=self.__alpha)
        self.__model.fit(features,labels)

    def classify(self,sentence):
        vec = self.__vectorizer.vectorize(sentence)
        prediction = self.__model.predict(vec)
        return prediction[0]

    def getParameters(self):
        return {"Alpha":self.__alpha}
