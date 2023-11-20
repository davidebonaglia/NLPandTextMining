# NPL and text mining

![Picture 1](https://github.com/davidebonaglia/NLPandTextMining/assets/75437147/1b3f9b34-0a5b-4fe0-8877-7b83f4f09508)

### Text pre-processing (tokenization, lemmatization, stemming, lowercase, …)

How do we represent a document? With VECTORS
And how do we represent vectors?
- Bag of words: vector space model where the terms are the words or n-grams, and the weights are the number of occurrences of the term in the document (binary, term frequency TF and term frequency inverse document frequency TFiDF).
- Topics
- Words embedding

### Text classification (text transformation: features generation)

Features: information which is used to separate data into classes. Features are used to predict labels of the data, which may be compared to known correct labels. 

Which features to use?
- Words (unigrams)
- Phrases/n-grams
- Sentences

In other words, by classifying text, we are aiming to assign one or more classes to a document, making it easier to build future prediction.

**Supervised learning** (i.e., learning from the labeled data). Classes are predefined and documents within the training dataset are manually tagged with one or more category labels. A classifier is then trained on the dataset, which means it can predict a new document’s classes never seen before.
**Unsupervised learning**: find structures in unlabeled data (e.g., clustering texts together based on similar metrics)
How to train a classifier?
- Split the data into training and test set (avoid overfitting)
- Validate the classifier by finding the most successful settings in the training set
- Test classifier on test set (unseen data)
- Estimate accuracy (i.e., percentage of correct labels).

Different classification strategies:
- Binary classification (e.g., predicted the age or gender of the reviewer based on the test of his review):
- Logistic regression 
- Support vector machine classifier (SVM)
- Multi-class classification (e.g., can we predict, based on review text, which genre (fiction, novel, thriller, …) the reviewer discusses?):
- K-nearest neighbor classifier (text are considered close neighbors if they share many words)
- Naïve Bayes classifier (calculates probabilities of different labels on each text, based on words in the text)
- Decision tree classifier (words features are used to separate classes (e.g., if the document contains “alien”, it is likely is a sci-fi))

How to optimize classifiers?
- Random forest classifier: fits multiple decision trees on subsets of the data
- Voting classifier


### Feature selection
Imagine that we have some data, and we want to build a classifier so that we can predict something. The data has 10,000 features though. We need to cut the features down to 1,000 before trying the training process. But which 1,000 feature should we chose? The process of choosing the 1,000 features is called Feature Selection.
Feature Selection, therefore, is a process of selecting an optimal subset of features of the training set according to a certain criterion and using just those selected features in the classification algorithm.

Why do we want to exploit Feature Selection? Mainly to increase the accuracy because the more features we have (terms, n-grams, sentences), the more accuracy decreases
- Reduce noise in feature representation
- Improve final classification performance
- Improve training/testing efficiency

Methods for feature selection:
![Picture 2](https://github.com/davidebonaglia/NLPandTextMining/assets/75437147/2b33b6c8-f7d5-4efa-8a95-b00bf90b6c20)

- Filter: specify some metrics and filter out the features based on those metrics. An example of filter-based feature selection could be the chi-square or the Gini index
- Wrapper: these methods try out different feature subsets and, by iterating different subset at each step, find the best subset to train our model.
- Embedded: these methods use algorithm that have built-in feature selection methods (for instance, Lasso and Ridge)

### Unsupervised learning: Clustering 
- Data is not labeled
- Group points that are close to each other could be clustered together
- Identify structure or patterns in data

**Hard clustering**: each document belongs to exactly one cluster
**Soft clustering**: a document can belong to more than one cluster

Clustering algorithms:
- Partitional clustering: construct a partition of n documents into a set of K clusters. An example of partitional clustering is the k-means algorithm. K-mean algorithm assumes documents are real-valued vectors. Clusters are based on centroids of points in a cluster
- Hierarchical clustering: it builds a tree-based hierarchical taxonomy (dendrogram) from a set of documents. Clustering is obtained by cutting the dendrogram at a desired level. 
Typical hierarchical clustering algorithms are bottom-up agglomerative clustering and top-down divisive clustering
- Topic modelling: documents are a collection of words and have a probability distribution over topics. Topics have a probability distribution over words. Topics made up of words are therefore used to generate documents

Cluster validation:
- Internal validation: we evaluate the coherence of the cluster algorithm.
- Inter-cluster similarity vs intra-cluster similarity
- Davis-Boudin index. The smaller the DB index, the better.
- External validation: the success of the cluster procedure is measured by its ability to discover some or all of the hidden patterns or latent classes in gold standard data

### Words embedding 
How can we represent the meaning of words? We can ask how similar is ‘cat’ with ‘dog’ or ‘Paris’ to ‘London’ (and its extension, how similar is Document A to Document B)?

Can we represent words as vectors? The vector representation should:
- Capture semantics (similar words should be close to each other in the vector space; relation between two vectors should reflect the relationship between the two words)
- Be efficient (vectors with fewer dimensions are easier to work with)
- Be interpretable

Distributional hypothesis: words that occur in similar contexts tend to have similar meanings

Words as vectors:
- One hot encoding: maps each word to a unique identifier (vectors representation: all dimensions are zeros except one that is equal to 1)
- Co-occurrences: two variants:
![Picture 3](https://github.com/davidebonaglia/NLPandTextMining/assets/75437147/c3735080-4b7c-421c-8cab-2878b4d2456a)

Words embeddings:
- Vectors are short (typically 50-124 dimensions)
- Vectors are dense (most non-zero values)
- Very effective for many NLP tasks

#### Word2vec
<img width="360" alt="Picture 4" src="https://github.com/davidebonaglia/NLPandTextMining/assets/75437147/9449f22e-553e-48ba-a640-8af3b62fcaeb">

Continuous Bag-of-Words VS Skip-gram
Both are architectures used to learn the underlying word representations for each word by using neural networks.
<img width="452" alt="Picture 5" src="https://github.com/davidebonaglia/NLPandTextMining/assets/75437147/c582bb66-dfd0-40ca-a6a2-59e0dce45ec1">

 CBOW: the distributed representations of context (or surrounding words) are combined to predict the word in the middle
- Skip-gram: the distributed representation of the input word is used to predict the context

Skip-gram:
![Picture 6](https://github.com/davidebonaglia/NLPandTextMining/assets/75437147/81cc583f-9de9-4c99-8416-f672d8f49ae1)

A word c is likely to occur near the target word if its embedding is similar to the target embedding (w*c). This turns into a probability to using a sigmoid function:
$$
P(+|w,c) = \frac{1}{1 + e^{-wc}}
$$
![Picture 7](https://github.com/davidebonaglia/NLPandTextMining/assets/75437147/dc98b106-c44c-45d3-b571-dee06a20396a)

We start with random embedding vectors.
During the training, we try to:
- Maximize the similarity between the embeddings of the target word and context words from the positive examples
- Minimize the similarity between the embeddings of the target words and context words from the negative examples
After the training:
- Frequent word-context pairs in the data: w*c high
- Not word-context pairs in data: w*c low

So, words occurring in same contexts are close to each other

<img width="452" alt="Picture 8" src="https://github.com/davidebonaglia/NLPandTextMining/assets/75437147/16bfbab4-e5ae-479d-b8bf-e93c2a58276c">

<img width="452" alt="Picture 9" src="https://github.com/davidebonaglia/NLPandTextMining/assets/75437147/dfa77429-c940-48e7-bf0b-afdaa2ac807a">





