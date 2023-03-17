---
layout: post
tags: data-science system-design search
comments: true
---

<!-- ---
title: 'Building a Machine Learning-Based Product Search Engine'
author: babaniyi
comments: true
date: 2023-03-11

tags:
- Data Science
- ML System design
- Search

On e-commerce sites like Amazon, there may be millions of different things to choose from. For instance, a customer on Amazon could type in "casual shirt for winter" and receive a list of available shirts. A system has been set up in the background that retrieves images related to the user's query, ranks them according to their similarity to the query, and then shows them to the user.

In this post, I describe how to create a search system that works in a manner similar to that of e-commerce platforms by taking a user's text query and returning a list of relevant photos.

--- -->

The number of products available on ecommerce platforms like Amazon, Alibaba, Ebay can be huge and can grow into hundred of thousands if not millions. For instance, on Amazon a user may input **casual shirt for summer**, then a list of shirts that are available are returned to the user. In simple terms, what happens in the background is that, a system has been put in place that retrieves images similar to the user's query, ranks them based on the similarities to the query and then displays them to the user. 

The system that does this hardlifting is my focus in this article, that is, I explain how to design a product search system similar to the one used by leading e-commerce platforms that takes a user's text query and returns a list of images that are relevant to the text query.


<figure>
    <img src="../images/blogs/img_search_text_query.png" 
         alt="Search image using text">
    <figcaption>Figure 1.1: Searching images with a text query</figcaption>
</figure>

## Basic requirements
To summarize the problem statement, we are designing an image search system that retrieves images similar to the user query, ranks them based on their similarities to the text query, and then displays them to the user. We make some basic assumptions which includes:

1. For simplicity, we assume a user can only input text queries (images or videos queries are not allowed).
1. We do not need to personalize the result of the search system. 
1. We have a dataset of say one million `<image, text query>` pairs for model training. 
1. The model uses image metadata and pixels. Also, users can only click on an image and we can construct training data online and label them based on user interactions.


# 1. Overview of the search system
As shown in Figure 1.2 below, the search system takes a text query as input and outputs a ranked list of images sorted by their relevance to the text query.

<figure>
    <img src="../images/blogs/img_search_text_input.png" 
         alt="Search input-output">
    <figcaption>Figure 1.2: Image search system's input-output</figcaption>
</figure>

This kind of problem is known as a ranking problem. In general, the goal of ranking problems is to rank a collection of items such as images, websites, products, etc., based on their relevance to a query, so that more relevant items appear higher in the search results. Many ML applications, such as recommendation systems, search engines, document retrieval, and online advertising, can be framed as ranking problems. 

In order to determine the relevance between an image and a text query, we utilize both image visual content and the image's textual data. An overview of the design can be seen in Figure 1.3.

<figure>
    <img src="../images/blogs/img_search_text_overview.png" 
         alt="Overview, search system">
    <figcaption>Figure 1.3: High-level overview of the search system</figcaption>
</figure>



## 1.1 Text search
Images that have titles, descriptions, or tags that are the most comparable to the text query are displayed as the results of the **text search**. When a user enters in a text query like *casual shirts for summer*, Figure 1.4 demonstrates how text search functions. Using full-text search in a database is one method to find the image description that matches the text query the most.

Full-text search (FTS) is the process of searching a portion of text within a large collection of electronically recorded text data and providing results that include part or all of the terms in the search [2]. It retrieves documents that don’t perfectly match the search criteria. Documents here refers to database entities containing textual data. 

You may enhance a FTS with search tools like fuzzy-text and synonyms in addition to looking for specific keywords. As a result, while searching with a phrase like "pasta," the results could also include "Fettuccine Carbonara" or "Bacon and pesto flatbread" in addition to dishes like "Pasta with meatballs". This implies that, for example, if a user searches for "cats and dogs," an application supported by FTS may be able to return results that include simply "cats" or "dogs," the words in a different order ("dogs and cats," or "cat"), or alternative spellings of the terms ("dog," or "cat") [3]. Applications can better infer the user's intent thanks to this, which speeds up the return of relevant results.
This technique is not based on machine-learning hence it is fast as there's no training cost involved. Many search engine companies use search engines such as Elasticsearch[4], MongoDB Atlas Search[2] to return the results of text queries similar to those you have available as seen in Figure 1.4.


<figure>
    <img src="../images/blogs/img_search_text_txt.png" 
         alt="Text search">
    <figcaption>Figure 1.4: Text search</figcaption>
</figure>


## 1.2. Image visual search
This component outputs a list of images after receiving a text query as input. Based on how closely the text query and the image resemble each other, the images are ranked. **Representation learning** is a method that is frequently used to do this.

In representation learning, a model is trained to turn data like images and texts into representations referred to as embeddings. Another way to describe it is that the model converts the images and text queries into points in the embedding space, an N-dimensional space. These embeddings are trained so that nearby embeddings in the embedding space exist for similar images [1]. 

It is important to note that in this approach, the text query and image are encoded separately using two encoders. 
First, we use a machine learning model which does the image encoding that generates an embedding vector for all the images of the products available. Figure 1.5 illustrates how similar images are mapped onto two points in close proximity within the embedding space.

Then, we use a text encoder that generates an embedding vector from the text. Finally, we calculate the similarity score between the image and text embedding using a dot product of their representation.

<figure>
    <img src="../images/blogs/img_search_text_embed.png" 
         alt="Image embedding example">
    <figcaption>Figure 1.5: Similar images in the embedding space</figcaption>
</figure>

# References
1. Ali Aminian & Alex Xu (2023). *Machine Learning System Design Interview*
2. Full Text Search with MongoDB. https://www.mongodb.com/basics/full-text-search
3. How To Improve Database Searches with Full-Text Search. https://www.digitalocean.com/community/tutorials/how-to-improve-database-searches-with-full-text-search-in-mysql-5-6-on-ubuntu-16-04
4. Elastic Search. https://www.elastic.co/elasticsearch/
