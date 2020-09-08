# Text Summarization Techniques
Text summarization is getting an idea of text in fewer words. In this we apply different techniques for text
summarization:

#### 1. Term-Frequency: This is the method of summarization based on the frequency of the words that are used in the text.
Idea:
* First method of constructing by taking each word and then assigning the frequency for that word as
score of the word.
* Second method is similar but in this we remove the stop words (words that does not contribute in the
meaning of sentence e.g. is, am, are) and special characters (e.g. ‘(‘, ’]’ ). This method performs better
than First One.

Steps:
* Load data. Perform sentence segmentation.
* Collect all words with their frequencies based on the method first or second.
* Score each sentence based on the frequency of each word in the sentences.
* Longer sentences are more likely to have greater score even they may not carry significant meaning.
Thus, we normalize the score by dividing it with sentence length.
* Set the threshold of the sentences by taking average score of all the sentences.
* Generate summary by considering only the sentences whose score is more than threshold.

#### 2. Term-Frequency Inverse-Document-Frequency (TF-IDF): It is extension of above method. In this method with the frequency of word, frequency of sentences relevant to words is also considered.

Idea:
* Consider two words having same frequency (e.g. 4) and not the part of stop words with the difference
that first word is used in only two sentences twice and second word is used four times in four sentences.
Now the word with four sentences has lower significance in the summarization than word with two
sentences. So, we reduce the word significance by multiplying it with inverse frequency of sentences in
which word lies.
Advantage:
* This method works well even without removal of stop words form the dataset, as the stop words are
used throughout the text and directly get penalized by IDF factor.

Steps:
* Load data. Perform sentence segmentation.
* Collect all words with their frequencies. (Removing stop words give better performance).
* Collect log (IDF) for each word.
* Prepare the word-set with all words having score as product of word frequency and IDF.
* Score sentences based on the score of all the words.
* Longer sentence tend to have advantage over smaller sentences. So, we normalize by dividing by the
number of words used in sentence.
* Threshold is found. Simplest way to do this is by calculating the average.
* Generate summary by checking whether score of sentence is greater than threshold or not.

#### 3. Centroid Based Clustering: This method uses similarity score to find the sentences having meaning similar to title of text, this work much better for the text with title. For text with no title, sentences with most relevant words tend to dominate. 

Idea:
* Sentence with similar words tend to have similar meaning. Thus, cosine matrix is used where it compare each word of one sentence to each word of other and provide similarity for complete sentence. 

Steps:
* Load data. Perform sentence segmentation.
* Collect all words. (Removing stop word).
* Tokenize all sentences with the relevant words.
* Convert all tokenized sentences into vectors.
* Find the cosine similarity matrix for each sentence.
* Compare all sentences with title or most relevant sentence of the text and generate summary.

#### 4. Scoring Indicator Representation: This method extract the features (like title words, sentence length etc.) and calculate score based on the features. This technique outperform all the three above approachs.

Idea: Instead of assigning score to words, assigning score to sentences are better way for summarization. 

Features:
* Title words: Sentence with the title words are more likely to contain information regarding article.
* Sentence length: The longer sentences are more likely to carry more information about the text. Hence we prefer longer sentences for summary consideration
* Sentence position within paragraph: Every beginning sentence of the article or paragraph either contain new idea or justification or explanation of the previous sentences mentioned. This implies the position of sentence held is important.
* Sentence similarity: Sentences with similar meaning are more important to represent a single idea. Thus, articles with four clusters having multiple sentences can be represented in only four sentences. Or in depth knowledge of particular type is required it can be explored by looking only all the sentences of cluster
* Term weight: This is the representation of words used in the sentences into numbers. Generally, a method from any of the above is used to represent weight of words. (E.g. TF-IDF).
* Numerical data: Sentences with numbers are likely to have importance as quantifying a data is much better representation than words (E.g. Company made Hugh profit this year vs. Company had a 50 million turnover more in 2017 than in year 2016).

#### Many more features can be added to increase the preformance

Advantage:
* This method has proven to be much better at performing summarization.
* The number of features does not change with the very with the length of article thus low memory is used.
* Model can be easily trained and modified.

Steps:
* Load data and perform sentence segmentation.
* Tokenize all sentences with the relevant words.
* Find features of all the sentences in the article and create score matirx for each feature.
* For calculating the score, we have two steps:
    Normalization: Limit each sentence into range (0,1) and calculate the score of all features based in the sentences.
    Fuzzy Set Logic: Divide the entire range of each feature into n equal parts and assign value for a fixed range.
    (Fuzzzy Set Logic seems to perform better than Normalization but correct threshold must be set for each freature)
* Generate the summary based in the scores.

#### 6. Machine Learning:

Idea:
In Machine Learning the summarization concept is modified to classification. There is a need for summary to perform classification over text. This summary can be obtained both by humans or any of the above method. Now, By training the model over existing summary, the testing is done and then for next articles with each sentences we get whether the sentence is part of the summary or not.

Steps:
* Convert all the sentences into vectors either by Indicator representation or TF-IDF or TF.
* A dependent variable is created whether a sentence is a part of summary or not.
* Create the classification model with Naïve Bayes or Logistic Regression or SVM.
* For any target convert the article in the vectors and run model.
* For each sentences the prediction equal 1 sentences are collected and then combined to form a summary.

#### 7. Deep Learning:
