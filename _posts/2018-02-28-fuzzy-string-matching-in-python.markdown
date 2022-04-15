---
layout: post
tags: data-science text-analysis nlp
comments: true
---

As a data scientist, you are forced to retrieve information from various sources by either leveraging publicly available API's, asking for data, or by simply scraping your own data from a web page. All this information is useful if we are able to combine it and not have any duplicates in the data. But how do we make sure that there are no duplicates?

<!-- more -->

I know ... _"duh! you can just use a function that retrieves all the unique _information thus removing duplicates"_. Well, that's one way, but our function probably can't tell that a name like _"Barack Obama"_ is the same as _"Barack H. Obama"_ right? (Assuming we were retrieving names of the most famous people in the world). We can clearly tell that these names are different but they are probably referring to the same person. So, how do we match these names?

This is where Fuzzy String Matching comes in. This post will explain what Fuzzy String Matching is together with its use cases and give examples using Python's Library [_Fuzzywuzzy_](https://pypi.python.org/pypi/fuzzywuzzy).


#### **Fuzzy Logic**




<blockquote>Fuzzy(adjective): difficult to perceive; indistinct or vague

-Wikipedia</blockquote>


[Fuzzy logic](https://en.wikipedia.org/wiki/Fuzzy_logic) is a form of multi-valued logic that deals with reasoning that is approximate rather than fixed and exact. Fuzzy logic values range between 1 and 0. i.e the value may range from completely true to completely false. In contrast, _**Boolean Logic**_ is a two-valued logic: true or false usually denoted 1 and 0 respectively, that deals with reasoning that is fixed and exact. Fuzzy logic tends to reflect how people think and attempts to model our decision making hence it is now leading to new intelligent systems(expert systems).

So, if we are comparing two strings using fuzzy logic, we would be trying to answer the question _"How similar are string A and string B?", and rephrasing it as "Are string A and String B the same?"_ when using the Boolean Logic.


#### Fuzzy String Matching


[Fuzzy String Matching](https://en.wikipedia.org/wiki/Approximate_string_matching), also known as Approximate String Matching, is the process of finding strings that approximately match a pattern. The process has various applications such as _spell-checking_, _DNA analysis and detection,_ spam detection, _plagiarism detection e.t.c_


#### Introduction to _Fuzzywuzzy_ in Python


**Fuzzywuzzy** is a python library that uses **Levenshtein Distance** to calculate the differences between sequences and patterns that was developed and also open-sourced by [SeatGeek,](https://seatgeek.com/) a service that finds events from all over the internet and showcase them on one platform. The big problem they were facing was the labeling of the same events as stated on their [blog](http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/). This is the same as the example I gave at the beginning of the post where an entity such as a person's name can be labelled differently on different sources.


##### Installation


To install the library, you can use pip:

```python
pip install fuzzywuzzy
pip install python-Levenshtein
```

##### Examples


First we have to import the fuzzywuzzy modules:

```python

from fuzzywuzzy import fuzz
from fuzzywuzzy import process
```

Now, we can get the similarity score of two strings by using the following methods; ratio() or partial_ratio():

```python

fuzz.ratio("Catherine M Gitau","Catherine Gitau")

91
```
```python
fuzz.partial_ratio("Catherine M. Gitau","Catherine Gitau")

100
```


You're probably wondering why the scores are different. This is because the fuzz.ratio() method just calculates the edit distance between some ordering of the token in both input strings using the `difflib.ratio.` You can find out more about the difflib.ratio [here](https://docs.python.org/2/library/difflib.html#difflib.SequenceMatcher.ratio). The **_fuzz.partial_ratio()_** takes in the shortest string, which in this case is "Catherine Gitau" (length 14) , then matches it with all the sub-strings of length(14) in "Catherine M. Gitau" which means matching with "Catherine Gitau" which gives 100%. You can play around with the strings until you get the gist.

What if we switched up two names in one string? In the following example, I've interchanged the name "Catherine Gitau" to "Gitau Catherine" .Let's see the scores:

```python

fuzz.ratio("Catherine M Gitau","Gitau Catherine")
55

fuzz.partial_ratio("Catherine M. Gitau","Gitau Catherine")
60

```

We see that both methods are giving out low scores, this can be rectified by using **_token_sort_ratio()_** method. This method attempts to account for similar strings that are out of order. Example, if we used the above strings:

```python
fuzz.token_sort_ratio("Catherine Gitau M.", "Gitau Catherine")
94

```

As you can see, we get a high score of 94.


#### Conclusion


This article has introduced Fuzzy String Matching which is a well known problem that is built on Leivenshtein Distance. From what we have seen, it calculates how similar two strings are. This can also be calculated by finding out the number of operations needed to transform one string to the other .e.g with the name "Barack", one might spell it as "Barac". Only one operation is needed to correct this i.e adding a K at the end. You can try this out using the _stringdist_ library in **r** as such:

```r

adist("Barack", "Barac")
[1]

```



#### Sources


[Fuzzy string matching in Python](https://marcobonzanini.com/2015/02/25/fuzzy-string-matching-in-python/)



Till next time:)


