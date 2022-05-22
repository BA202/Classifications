import transformers
import tensorflow as tf
import numpy as np


class BERT:

    def __init__(self,trainingData=None,debug = False,**kwargs):
        self.__modelName = 'bert-base-uncased'
        self.__maxLength = 100
        self.__trainingEpochs = 10
        self.__learningRate=5e-7
        self.__validationSplit = 0.2

        self.__catToInt = {cat:i for i,cat in enumerate(list({sample[1] for sample in trainingData}))}
        self.__intToCat = {self.__catToInt[key]: key for key in self.__catToInt.keys()}

        self.__model = transformers.TFAutoModelForSequenceClassification.from_pretrained(self.__modelName, from_pt=True, num_labels=2)
        self.__tokenizer = transformers.AutoTokenizer.from_pretrained(self.__modelName)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.__learningRate, epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

        tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir="logs")
        earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        self.__model.compile(optimizer=optimizer, loss=loss,metrics=metrics)
        self.__model.summary()

        sentencesAsVec = self.__tokenizer(
            [sample[0] for sample in trainingData],
            add_special_tokens=True,
            max_length=self.__maxLength,
            truncation=True,
            padding="max_length",
            return_tensors="tf",
            return_token_type_ids=False,
            return_attention_mask=True,
            verbose=True,
        )

        self.__model.fit(
            x=sentencesAsVec['input_ids'],
            y=np.array([self.__catToInt[sample[1]] for sample in trainingData]),
            validation_split=self.__validationSplit,
            batch_size=64,
            epochs=self.__trainingEpochs,
            callbacks=[tensorboardCallback,earlyStoppingCallback]
        )

    def classify(self,sentence):
        sentenceAsVec = self.__tokenizer(
            [sentence],
            add_special_tokens=True,
            max_length=self.__maxLength,
            truncation=True,
            padding="max_length",
            return_tensors="tf",
            return_token_type_ids=False,
            return_attention_mask=True,
            verbose=True,)
        prediction = self.__model(sentenceAsVec)[0][0]
        prediction = tf.nn.softmax(prediction,axis=-1)
        if prediction[0] > prediction[1]:
            return self.__intToCat[0]
        else:
            return self.__intToCat[1]


    def getParameters(self):
        return {'Base Model Name':self.__modelName ,'Training Epochs':self.__trainingEpochs,'Learning Rate':self.__learningRate,'validation Split':self.__validationSplit}
