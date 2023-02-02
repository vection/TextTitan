
![White logo - no background](https://user-images.githubusercontent.com/28596354/216412245-49870702-05e5-46fe-a322-9233c6f97d47.png)


TextTitan library offers a comprehensive set of tools for tackling a wide range of NLP tasks including classification, regression, multi-modal and generation tasks.


** Still under development

## Introduction
TextTitan is a comprehensive NLP library that provides end-to-end solutions for a wide range of NLP tasks. This library makes it easy for developers and researchers to perform complex NLP tasks with ease and efficiency.

With TextTitan, you can tackle classification, regression, multi-modal and generation tasks with fast and accurate results. The library has been designed to be production-ready, so you can quickly integrate NLP capabilities into your projects and start seeing results.

Whether you're a beginner or an experienced NLP practitioner, TextTitan provides an intuitive and easy-to-use interface that makes NLP accessible to everyone. So why wait? Start using TextTitan today and revolutionize your NLP workflow!

### Installation
Soon 


## Quick Usage

Models supported: Bert, Roberta, DebertaV3, LSTM, LSTM+CNN 


```
classifier = NLPClassifier(base_model='lstm',problem_type='single_label_classification',save_path='best_weights')
classifier.max_length = 64
```

To train the model you need to provide:
- CustomDataset 
- Array

```
text_list = df['OriginalTweet'].tolist()
train_label_list =df['Sentiment'].tolist()
classifier.train(text_list,train_label_list,epochs=15,batch_size=128)
```

TextTitan automatically saves the best weights according to validation set with early stopping.


For more information reach Documentation


### Loading & Evaluation

Let assume we trained model on default name - best_weights.

** If its Bert / Roberta models it will contain only one folder.

** If its LSTM/LSTM+CNN weights file will be best_weights.pth and best_weights_tokeniezr for tokenizer

We always follow by model name in loading function. 

```
classifier = NLPClassifier.load('best_weights.pth')
classifier.predict(['test1 test test ','test2 test test']

[('Positive', 0.27153775095939636), ('Positive', 0.30906566977500916)]
```






 
