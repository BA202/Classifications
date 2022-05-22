from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from StatisticalMLVectorizer import StatisticalMLVectorizer

class LogisticRegression:

    def __init__(self,trainingData= None, kernel = 'linear',debug = False,**kwargs):
        self.__vectorizer = StatisticalMLVectorizer()

        features = self.__vectorizer.vectorize(
            [sample[0] for sample in trainingData])
        labels = [sample[1] for sample in trainingData]

        gridSearchParmeters = {'class_weight': ["balanced"],
                               'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                               'fit_intercept': [True,False]}

        grid_search = GridSearchCV(skLogisticRegression(),
                                   gridSearchParmeters,
                                   cv=5, return_train_score=True,
                                   n_jobs=-1)
        grid_search.fit(features, labels)
        print("best param are {}".format(grid_search.best_params_))
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        for mean, std, param in zip(means, stds,
                                    grid_search.cv_results_['params']):
            print("{} (+/-) {} for {}".format(round(mean, 3), round(std, 2),
                                              param))
        self.__model = skLogisticRegression(
            class_weight=grid_search.best_params_['class_weight'],
            penalty=grid_search.best_params_['penalty'],
        fit_intercept=grid_search.best_params_['fit_intercept'])
        self.__model.fit(features, labels)

    def classify(self,sentence):
        vec = self.__vectorizer.vectorize(sentence)
        prediction = self.__model.predict(vec)
        return prediction[0]


    def getParameters(self):
        modelParams = self.__model.get_params()
        return {'penalty': modelParams['penalty'],
                'fit_intercept': modelParams['fit_intercept']}
