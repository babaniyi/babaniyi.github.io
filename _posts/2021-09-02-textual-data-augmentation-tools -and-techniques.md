---
layout: post
tags: nlp data-augmentation
comments: true
---
{:.no_toc}

**Data Augmentation** is a technique that's used to increase the amount of available data by modifying slightly already existing data. One of the limitations of NLP especially for low-resourced languages, is the unavailability of labelled data and it usually takes a great deal of money and time to manually annotate/label the relevant datasets. Therefore, it's become necessary to come up with methods of automatically increasing the available data so as to improve the performance of the model.

Data augmentation is mostly popular in the computer vision domain where it involves generating augmentations using simple image transformations such as rotating or flipping the images with the aim of increasing training data for the model. However, it is very challenging to find the appropriate methods for text data augmentation because language in itself is complex and it can be difficult to preserve the context, grammar and semantics.

In this article, we'll be looking at how one can use some available data augmentation methods and tools to augment text that's low-resourced. A lot of the tools available for data augmentation are mostly used for the English language. In this article, I present to you some of the methods that can be used for languages other than English but the article will focus on the KiSwahili language. 

Overall, some of the popular methods I will be discussing are:

- Lexical substitution
- Random noise injection

# Lexical Substitution

This is the task of identifying a substitute for a word in a sentence without changing the meaning of the sentence. There are various techniques that can be used for this:

## i) Fasttext-based augmentation
In this technique, we use the pre-trained word embedding Word2Vec and use the nearest neighbor words in the embedding space as a replacement for a particular word in the sentence.

For example, you can replace a word with 3 most similar words and get three variations of the text while preserving its context.

In this example, I will use fasttext Embeddings which has pre-trained models for over 100 different languages which you can find [here](https://fasttext.cc/docs/en/crawl-vectors.html). I will also be using the [Textaugment](https://github.com/dsfsi/textaugment) library which is used for augmenting text for natural language processing applications.


```python
#import the necessary libraries
! pip -q install textaugment gensim
import textaugment gensim

# Download the the FastText embeddings in the language of your choice in this case I'm downloading Swahili
!wget "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sw.300.bin.gz"

# save path to your pre-trained model
from gensim.test.utils import datapath
pretrained_path = datapath('./cc.sw.300.bin.gz')

# load the model
model = gensim.models.fasttext.load_facebook_model(pretrained_path)

from textaugment import Word2vec
t = Word2vec(model = model.wv)
output = t.augment('John anacheza mpira')
>> john anacheza mipira

```


## ii) MLM augmentation

MLM or Masked Language Modeling is the task of learning a model to predict missing tokens that have been masked based on its context. Transformer models like BERT, ALBERT and ROBERTA have been trained on large amounts of text using this task.

This method can also be used in augmenting some texts by using a pre-trained transformer model like BERT, mask some words in the sentence and use the BERT model to predict the token that's been masked.

So, we can generate a number of sentences using the predictions made by the model. This particular method produces sentences that are more grammatically correct because the model takes into account the context when making te predictions.

For this task you could use the [Multilingual BERT base model](https://github.com/google-research/bert/blob/master/multilingual.md) or [XLM-ROBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta) which was trained on 100 different languages.

## iii) TF-IDF based augmentation
TF-IDF also known as Term Frequency-Inverse document frequency tells us how uninformative a word is. So, low TF-IDF scores means that a word is uninformative and thus can be replaced without affecting the ground truth labels of the sentence.

For this particular task, you could use the [nlpaug library](https://github.com/makcedward/nlpaug). Have a look at this [notebook](https://github.com/makcedward/nlpaug/blob/master/example/tfidf-train_model.ipynb) to see an example of how you can use the library.



# Random noise injection

The idea behind this is to inject some noise in the text. For these particular tasks we'll use the [textaugment library](https://github.com/dsfsi/textaugment)

## i) Random Deletion
```python
from textaugment import EDA
t = EDA()
t.random_deletion('John anacheza mpira kwenye uwanja wa Nyayo')
>> anacheza mpira kwenye uwanja wa Nyayo
```

## ii) Random swap

```python
from textaugment import EDA
t = EDA()
t.random_swap('John anacheza mpira kwenye uwanja wa Nyayo')
>> John Nyayo mpira kwenye uwanja wa anacheza

```

## Other Data Augmentation techniques

Other than the techniques I have mentioned above, there are others that you can look into:
- Back translation
    - In this method, machine translation models are used to paraphrase text while retaining its original meaning. The process involves taking a language (eg. in Swahili) and translating it to another language (eg.English) using the MT model. Then you translate the English sentence back into Swahili.

    The only downside to this method is that it is expensive and time consuming especially if you'd like to back-translate a lot of text.

- WordNet-based augmentation
    - This is another example of lexical substitution that involves replacing a word in a text with it's synonym. The most popular open-sourced lexical database for the English language is WordNet. I did not include this as one of the examples because I couldn't find any database for Swahili that's similar to WordNet. Please let me know in the comment section if you have a resource like WordNet that's open-source for low-resourced languages. 


# Conclusion

In this article I have introduced techniques and tools that can be used in data augmentation for text data specifically those that can be used for low-resourced languages. A lot of the pre-trained models are still not of good quality so for methods such as MLM-augmentation and Word2Vec, you might once in a while get some nonsensical words. This means that there's still a lot of work that needs to be done on that front.

Please let me know in the comments which other data augmentation methods you've used for a low resourced language.




## References
- [A visual Survey of Data Augmentation in NLP](https://amitness.com/2020/05/data-augmentation-for-nlp/)
- [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/pdf/1901.11196.pdf)
- [Improving short text classification through global augmentation methods](https://github.com/dsfsi/textaugment)

