import random
import traceback
import json
from DataHandler.DataHandler import  DataHandler
from ModelReport.ModelReport import ModelReport


from  Models.BERT import BERT
#from  Models.KNearestNeighbors import KNearestNeighbors
#from  Models.LogisticRegression import LogisticRegression
from  Models.MultinomialNaiveBayes import MultinomialNaiveBayes
#from  Models.RandomForest import RandomForest
#from  Models.SupportVectorMachine import SupportVectorMachine


class testConstants:
    folds = 10
    seed = 4.83819
    dataLocation = ""
    balancedDataSet = False
    balancedSplitDataSet = True

    modelsToEvaluate = [
        {
            'data': 'Score',
            'model': BERT,
            'modelName': "BERT",
            'modelCreator': "Tobias Rothlin",
            'mlPrinciple': "Transformers",
            'refrences': {
                'NultinomialNB Explained': "https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664",
            },
            'algorithemDescription': """""",
            'graphicPath': "",
            'graphicDescription': "",
            'dataSet': f"ClassifiedDataSetV1.3 with {folds} folds cross validation",
            'seed': seed,
            'kfolds': folds,
            'opParams': None
        },
        {
            'data': 'Score',
            'model': MultinomialNaiveBayes,
            'modelName': "MultinomialNaiveBayes",
            'modelCreator': "Tobias Rothlin",
            'mlPrinciple': "Transformers",
            'refrences': {
                'NultinomialNB Explained': "https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664",
            },
            'algorithemDescription': """""",
            'graphicPath': "",
            'graphicDescription': "",
            'dataSet': f"ClassifiedDataSetV1.3 with {folds} folds cross validation",
            'seed': seed,
            'kfolds': folds,
            'opParams': None
        }
    ]



def modelPerofrmaceEvaluation(data,model,modelName,modelCreator,mlPrinciple,refrences,algorithemDescription,graphicPath,graphicDescription,dataSet,seed,kfolds,opParams):
    if opParams == None:
        opParams = [None]

    for param in opParams:
        random.seed(seed)
        random.shuffle(data)
        if param == None:
             paramForModelName = ""
        else:
            paramForModelName = "_" + str(param)
            print(print(f"{str(param):.^100s}"))


        modelPerformanceReport = ModelReport(modelName+paramForModelName, modelCreator,
                                             mlPrinciple, refrences,
                                             algorithemDescription,
                                             graphicPath, graphicDescription,
                                             dataSet, str(seed))

        for k in range(kfolds):
            testDataStart = int(k * len(data) / kfolds)
            testDataEnd = int(k * len(data) / kfolds) + int(
                len(data) / kfolds)
            testData = data[testDataStart:testDataEnd]
            trainingData = []
            for element, i in zip(data, range(len(data))):
                if not (i >= testDataStart and i < testDataEnd):
                    trainingData.append(element)

            print(
                f"{k}-training({len(trainingData)}/{(len(trainingData) / (len(trainingData) + len(testData))) * 100:.2f}%):Test({len(testData)}/{(len(testData) / (len(trainingData) + len(testData))) * 100:.2f}%) Split completed")

            if testConstants.balancedSplitDataSet:
                trainingData = DataHandler.balanceDataSet(trainingData)

            modelPerformanceReport.addTrainingSet(trainingData)

            print(f"{k}-added training split to performance raport")
            myScoreClassifier = model(trainingData,param=param,debug=True)
            print(f"{k}-model has been trained with training set")
            testResults = []
            trainingResults = []

            for testCase in testData:
                testResults.append(
                    [testCase[1], myScoreClassifier.classify(testCase[0])])

            for testCase in trainingData:
                trainingResults.append(
                    [testCase[1], myScoreClassifier.classify(testCase[0])])

            print(f"{k}-model has been tested\n")
            modelPerformanceReport.addTestResults(testResults)
            modelPerformanceReport.addTrainingResults(trainingResults,myScoreClassifier.getParameters())

        print(f" -creating the model raport")
        modelPerformanceReport.createRaport(modelName+paramForModelName)


if __name__ == '__main__':
    testbenchDataHabler = DataHandler(testConstants.dataLocation,lan="English")
    #loops through the testConstants dict
    for model in testConstants.modelsToEvaluate[1:]:
        print("-Loading dataset:")
        if model['data'] == "Score":
            testData = testbenchDataHabler.getScoreData(testConstants.balancedDataSet)
            testData = [data for data in testData if not data[1] == "Neutral"]
        elif model['data'] == "Category":
            testData = testbenchDataHabler.getCategorieData("Location",testConstants.balancedDataSet)
        else:
            testData = []
            print("-Data Source not found!:")
            break

        try:
            print(f"{'Evaluating Model '+ model['modelName']:-^100s}")
            modelPerofrmaceEvaluation(testData,model['model'],model['modelName'],model['modelCreator'],model['mlPrinciple'],model['refrences'],model['algorithemDescription'],model['graphicPath'],model['graphicDescription'],model['dataSet'],model['seed'],model['kfolds'],model['opParams'])
            print(f"\u001b[32m{'Done Evaluating Model '+ model['modelName']:-^100s}")
            print("\u001b[0m")
            print(100*"-")
        except Exception as e:
            print(f"\u001b[31m{'Error During Testing!!':-^100s}")
            print(traceback.format_exc())
            print("\u001b[0m")