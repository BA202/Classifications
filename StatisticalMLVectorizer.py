import numpy as np
import transformers
from tensorflow.keras.utils import to_categorical

class StatisticalMLVectorizer:

    def __init__(self, tokenizerModelName = 'bert-base-uncased',useBertEmbdeding = False):
        self.__modelName = tokenizerModelName
        self.__max_length = 100
        self.__useBertEmbdeding = useBertEmbdeding

        self.__tokenizer = transformers.AutoTokenizer.from_pretrained(self.__modelName)
        self.__vectorizer = transformers.TFAutoModel.from_pretrained(self.__modelName, from_pt=True)

    def vectorize(self,sentence):
        toc = self.__tokenizer(text= sentence,add_special_tokens=True,max_length=self.__max_length,truncation=True,padding='max_length',return_tensors='tf',return_token_type_ids=False,return_attention_mask=False,verbose=True)
        if self.__useBertEmbdeding:
            vec = self.__vectorizer(toc).last_hidden_state[:, 0]
            vec = vec.numpy()
            return vec
        else:
            vec = np.array([sum(to_categorical(sample, num_classes=len(
                self.__tokenizer.get_vocab())))[1:] for sample in
                      toc['input_ids']])
            return vec

if __name__ == '__main__':
    myVectorizer = StatisticalMLVectorizer()
    print(myVectorizer.vectorize(["This is an example.","This is the second example"]))