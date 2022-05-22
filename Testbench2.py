import random
import traceback
import json
from DataHandler.DataHandler import  DataHandler
from ModelReport.ModelReport import ModelReport


from  Models.BERT import BERT
#from  Models.KNearestNeighbors import KNearestNeighbors
#from  Models.LogisticRegression import LogisticRegression
#from  Models.MultinomialNaiveBayes import MultinomialNaiveBayes
#from  Models.RandomForest import RandomForest
#from  Models.SupportVectorMachine import SupportVectorMachine


class testConstants:
    folds = 10
    seed = 4.83819
    dataLocation = ""
    balancedDataSet = False
    balancedSplitDataSet = False

    modelsToEvaluate = [
        {
            'data': 'Score',
            'model': BERT,
            'modelName': "MultinomialNaiveBayesOnScore",
            'modelCreator': "Tobias Rothlin",
            'mlPrinciple': "Multinomial Naive Bayes",
            'refrences': {
                'NultinomialNB Explained': "https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664",
                'Stanford NLP Course': "http://spark-public.s3.amazonaws.com/nlp/slides/naivebayes.pdf",
                'Stanford NLP Lecture': "https://www.youtube.com/playlist?list=PLLssT5z_DsK8HbD2sPcUIDfQ7zmBarMYv",
                'Engilsh Stopwords': "https://www.tutorialspoint.com/python_text_processing/python_remove_stopwords.htm"
            },
            'algorithemDescription': """The learning algorithm used in this classification is the Multinomial Naïve Bayes. This approach was chosen as it is easy to implement and is computational very efficient. The first step in the classification pipeline is removing all strop words for example 'i', 'me', 'my', 'myself', etc. A list of English stop word is provided by the nltk module. The stop words remover just removes every word that is in the list of stop words. Next the sentence is passed through the stemmer. Stemmers remove morphological affixes from words, leaving only the word stem. This is done with the PorterStemmer class from the nltk module. The final preprocessing step is to vectorize the sentence. This results in a bag of words representation of the sentence. First all the words must be tokenized and then counted. The result will be a numerical feature vector. To generate this vector the CountVectorizer class from sklearn is used.  This class implements both tokenization and occurrence counting in a single class. With the sentence now represented in a vector the Naïve Bayes classifier can work with this vector. For the implementation of the Naïve Bayes classifier the MultinomialNB class (sklearn) is used. """,
            'graphicPath': "/Users/tobiasrothlin/Documents/BachelorArbeit/ScoreClassifier/OverviewImg.png",
            'graphicDescription': "Classification Pipeline",
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
    for model in testConstants.modelsToEvaluate:
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