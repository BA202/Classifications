from sklearn import svm
from sklearn.model_selection import GridSearchCV
from StatisticalMLVectorizer import StatisticalMLVectorizer


class SupportVectorMachine:

    def __init__(self,trainingData= None, kernel = 'rbf',debug = False,**kwargs):
        self.__vectorizer = StatisticalMLVectorizer()

        features = self.__vectorizer.vectorize(
            [sample[0] for sample in trainingData])
        labels = [sample[1] for sample in trainingData]

        C_range = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        gamma_range = [1,1.2,1.3,1.4,1.5,1.6,1.7]
        degree = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        gridSearchParmeters = dict(gamma=gamma_range, C=C_range,kernel = [kernel],degree= degree,class_weight=['balanced'])
        grid_search = GridSearchCV(svm.SVC(),
                                   gridSearchParmeters,
                                   cv=10, return_train_score=True,
                                   n_jobs=-1)
        grid_search.fit(features, labels)

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
        self.__model.fit(features, labels)


    def classify(self,sentence):
        vec = self.__vectorizer.vectorize(sentence)
        prediction = self.__model.predict(vec)
        return prediction[0]


    def getParameters(self):
        modelParams = self.__model.get_params()
        return {'kernel':modelParams['kernel'],'degree':modelParams['degree'],'gamma':modelParams['gamma'],'C':modelParams['C'],'max_iter':modelParams['max_iter']}
