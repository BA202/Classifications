from StatisticalMLVectorizer import  StatisticalMLVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighbors:

    def __init__(self,trainingData= None,debug = False,**kwargs):
        self.__vectorizer = StatisticalMLVectorizer()

        features = self.__vectorizer.vectorize(
            [sample[0] for sample in trainingData])
        labels = [sample[1] for sample in trainingData]

        gridSearchParmeters = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10],
                               'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}

        grid_search = GridSearchCV(KNeighborsClassifier(),
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
        self.__model = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'],algorithm=grid_search.best_params_['algorithm'])
        self.__model.fit(features, labels)

    def classify(self,sentence):
        vec = self.__vectorizer.vectorize(sentence)
        prediction = self.__model.predict(vec)
        return prediction[0]

    def getParameters(self):
        modelParams = self.__model.get_params()
        return {'n_neighbors':modelParams['n_neighbors'],'algorithm':modelParams['algorithm']}
