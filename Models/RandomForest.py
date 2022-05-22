from StatisticalMLVectorizer import StatisticalMLVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

class RandomForest:

    def __init__(self,trainingData= None,**kwargs):
        self.__vectorizer = StatisticalMLVectorizer()

        features = self.__vectorizer.vectorize(
            [sample[0] for sample in trainingData])
        labels = [sample[1] for sample in trainingData]

        gridSearchParmeters = {'max_features': ('auto','sqrt'),
         'n_estimators': [500, 1000, 1500],
         'max_depth': [5, 10, None],
         'min_samples_split': [5, 10, 15],
         'min_samples_leaf': [1, 2, 5, 10]}
        grid_search = GridSearchCV(RandomForestClassifier(), gridSearchParmeters,
                                   cv=5, return_train_score=True,
                                   n_jobs=-1)
        grid_search.fit(features,labels)
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
            min_samples_leaf=grid_search.best_params_['min_samples_leaf'])
        self.__model.fit(features, labels)


    def classify(self,sentence):
        vec = self.__vectorizer.vectorize(sentence)
        prediction = self.__model.predict(vec)
        return prediction[0]


    def getParameters(self):
        modelParams = self.__model.get_params()
        return {'max_features': modelParams['max_features'],
                'max_depth': modelParams['max_depth'], 'n_estimators': modelParams['n_estimators'],
                'min_samples_split': modelParams['min_samples_split'], 'min_samples_leaf': modelParams['min_samples_leaf']}
