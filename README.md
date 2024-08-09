# Task

NLP (Natural Language Processing) includes tasks like regression, classification, and generation. A key challenge in NLP is figuring out "how to convert text into numbers" so machines can understand and process it.

I have used various state-of-the-art models to solve this task like LSTM, BILSTM, Glove, CBOW, Skipgram, SBERT and referred various blogs and internet content to understand the concepts, datasets and models used in this task.

## Directory Structure


```
├── README.md
├── bertscore_slides.pdf
├── programming_task.md
├── Dataset_Analysis
│   ├── paws.ipynb
│   ├── PiC.ipynb
│   └── simlex.ipynb
├── Part_A
│   ├── cbow
│   ├── skipgram
│   ├── pre-trained models
│   ├── data
│   └── svd
├── Part_B
│   ├── Models
│   ├── Plots
│   ├── Sentence Similarity
│   └── Phrase Similarity
├── Bonus

```

## Word Similarity

For this task, I have coded three models from scratch representing words in vectors: CBOW, Skipgram, and SVD.
As mentioned in the doc, all these models are trained on unsupervised data.

Apart from these models, I have also used pre-trained models: GloVe and FastText as a comparative study.

1. **CBOW** : 
   - Continuous Bag of Words model  predicts the target word based on the context words. The context words are the words present in the window of the target word. The model is trained on the text corpus and the embeddings are obtained. 
    - The input to the model contains of the context words and the target word. The context and target words are separated from the input. The context consists of all words except the last one, and the target is the last word in each sequence.
    - The context words are passed through the embedding layer to obtain their embeddings, and then the mean of these embeddings is computed to represent the context. Similarly, the target word is passed through the embedding layer to obtain its embedding.
    - A score is calculated by taking the dot product of the context embedding and target embedding. This score is then passed through a sigmoid function to obtain a probability between 0 to 1.
    - The loss is calculated using the binary cross entropy loss function with the output label which is 0/1. The loss is then backpropagated to update the weights of the embedding layer.
    - Negative sampling has been implemented to enhance learning.


2. **Skipgram** :
    - Skipgram model predicts the context words based on the target word. The target word is the word  present in the center of the window. The model is trained on the text corpus and the embeddings are obtained. 
    - The input to the model contains of the target word and the context words. The target word is separated from the input. The context consists of all words except the target word.
    - The target word is passed through the embedding layer to obtain its embedding. Similarly, the context words are passed through the embedding layer to obtain their embeddings. Dot product is taken between the target embedding and each of the context embeddings. The dot products are summed across the context length to get a total score which is then passed through a sigmoid function to obtain a probability between 0 to 1.
    - The loss is calculated using the binary cross entropy loss function with the output label which is 0/1. The loss is then backpropagated to update the weights of the embedding layer.
    - Negative sampling is done using a unigram table. Each negative sample is constructed by replacing the target word with a word randomly chosen from the unigram table (excluding the target word). The unigram table is constructed by raising the frequency of each word to the power of 3/4 and then normalizing it. The unigram table is used to sample negative words with a probability proportional to their frequency.

3. **SVD** :
    - The co-occurence matrix is created where the element at position (i, j) in the matrix indicates the number of times word i occurs in the context of word j within a specified window size.
    - The co-occurence matrix is then factorized using Singular Value Decomposition (SVD) to obtain the embeddings of the words The factorization process effectively uncovers the latent structures in the co-occurrence matrix, encapsulating the essential features of the data.
    - The optimal embedding dimension is calculated by selecting the variance threshold.
    - The top k similar words to a given word are found by calculating the cosine similarity between the word and all the other words in the vocabulary. The top k words with the highest cosine similarity are returned.

### Preprocessing

I have trained the models on two text corpora separately: **wikitext-2-v1** and **English Word Dataset Synonyms and Antonyms** respectively.

The Wikitext corpus was selected because it's a collection of over 2M tokens of English text extracted from the set of verified articles on Wikipedia. Thus, the corpus suits the requirements of the task.

And, the English Word Dataset Synonyms and Antonyms was selected because the test dataset: SimLex-999 seems to have more of synonym-antonym pairs. Thus, I believed that training the models on this corpus would help in better performance on the test dataset.

Sentences were extracted from the corpus and then preprocessed and a vocabulary was created.

### Testing

The pretrained embeddings were used to find the similarity between the pair of words. The similarity was calculated using the cosine similarity between the embeddings of the words, and since the scores in the dataset are between 0-10.

The similarity scores were then compared with the actual simlex scores from the test dataset to calculate Spearman's correlation coefficient.

### Results

The performance on the SimLex-999 dataset was suboptimal. Even advanced pre-trained models such as GloVe and FastText struggled.

Models like CBOW and Skipgram tend to capture word associations as they derive a word's embedding from its surrounding words. Consequently, these models often fail to distinguish between words that are associated but not inherently similar.

Thus, I have tested my models on the WordSim-353 dataset as it was mentioned too in the paper.

The results are as follows:
<!-- Table -->
| Model | SimLex-999 | WordSim-353 |
| :---: | :---: | :---: |
| cbow_wiki | 0.06 | 0.09 |
| cbow_sa | 0.01 | - |
| skipgram_wiki | 0.08 | 0.03 |
| skipgram_sa | 0.17 | - |
| GloVe | 0.23 | 0.54 |
| FastText | 0.31 | 0.69 |

**Note** - All values are Spearman's correlation coefficient.

We observe that there is an improved performance on the WordSim-353 dataset.

However, the models I have trained from scratch do not perform as well as the pre-trained models. This is because the models are trained on a small subset of the corpus, which is not even 1% of training data used to train the actual models(~1000M tokens). Thus, the results are not so good.

### Additional Experiments and Observations

1. Supervised Training:
    - I have tried to build a supervised model on top of the pretrained models to improve the performance on the SimLex-999 dataset.
    - Basically, the architecture contains an MLP layer on top of the pre-trained model. The pretrained model is used to get the embeddings of the pair of words. The embeddings are concatenated and passed through the MLP layer to get the final output.
    - The loss is calculated using MSE loss with the Simlex scores.
    - However, this approach did not yield stable results and often resulted in overfitting.
    - The reason might be that we are not changing the embeddings of the pre-trained and are just training a linear layer on top of it. This might not be sufficient to capture the similarity between the words, and the size of the dataset is very small (1000), which might be another reason for the poor performance.

2. Checking similarity based on POS (Part of Speech) tags:
    - As mentioned in the paper, I tried to check the similarity between the words based on their POS tags. I used the FastText model for this experiment.
    - I have taken subsets of the SimLex-999 dataset based on the POS tags of the words. I have taken the following POS tags: Noun, Verb, and Adjective.
    - The results are as follows:
    <!-- Table -->
    | Model | Noun | Verb | Adjective |
    | :---: | :---: | :---: | :---: |
    | FastText | 0.496 | 0.258 | 0.578 |
    - The model performs better on the Adjective subset as compared to the Noun and Verb subsets.

3. Unkown words in vocab:
    - In the CBOW and Skipgram models, I have calcualted the number of word pairs in the Simlex dataset which are not present in the vocabulary of the model, and 128/1000 word pairs are not present in the vocabulary of the CBOW model. This adds to the poor performance of the model.


<!-- Word Similarity Over -->

## Sentence Similarity

In this task, I used a total of 4 models: LSTM, BILSTM, Average of Word Embeddings, SBERT.

I have used pretrained Glove embeddings (Glove-6B) for the first 3 models. For SBERT, I have used the pretrained model from the sentence-transformers library(bert-base-nli-mean-tokens).

1. **LSTM** :
    - The input to the model is a sentence, it is tokenized and padded to a fixed length. 
    - The embeddings of the words are obtained from the Glove embeddings these embeddings are packed and passed through the LSTM layer.
    - The output of the LSTM layer(last hidden state) is obtained for each sentence. These outputs are concatenated and passed through a linear layer. The output of the linear layer is passed through a sigmoid function to get a probability between 0 to 1.
    - The loss is calculated using the binary cross entropy loss function with the output label which is 0/1. The loss is then backpropagated to update the weights of the embedding layer.
2. **BILSTM** :
    - Same as LSTM, except that the LSTM layer is bidirectional.
    - The last 2 hidden states of the LSTM layer are concatenated and returned. This captures the context of the sentence in both directions.
3. **Average of Word Embeddings** :
    - The input to the model is a sentence. The sentence is tokenized and padded to a fixed length.
    - The average of word embeddings is taken for each sentence. The output embeddings of both sentences are concatenated and passed through a linear layer. The output of the linear layer is passed through a sigmoid function to get a probability between 0 to 1.
    - Loss function is the same as above.

### Training and Testing

Preprocessing is done for the sentences in the same way as it was done for the words.

The models are trained and tested on two datasets: **PAWS**. 

### Results

The results are poor on the PAWS dataset. This is because the PAWS dataset explicitly contains sentence pairs that have high lexical overlap without being paraphrased.
State of-the-art models trained on existing datasets have dismal performance on PAWS (<40% accuracy), as mentioned in the PAWS paper:

The results are as follows:
<!-- Table -->
| Model | PAWS | 
| :---: | :---: | 
| LSTM | 55.8% | 
| BILSTM | 56.2% | 
| Average of Word Embeddings | 55.4% | 
| SBERT-FineTuned | 75.54% | 


The BiLSTM model performs a bit better than the LSTM model because it captures the context of the sentence in both the directions. The accuracy of BiLSTM model is 56.2% on the PAWS dataset, which is very close the accuracy of the BILSTM model mentioned in the paper (57.6%).

The average of word embeddings model performs decently well despite just takes the average of the word embeddings of the sentence.

After fine tuning on 15k sentences from the PAWS train dataset, the SBERT model has an accuracy of 75.54% on the PAWS test dataset. Only 15/50k sentences were taken for fine tuning as the training time was very high.

### Additional Experiments

1. Cosine Similarity vs Linear Layer:
    - I have tried to replace the linear layer in the architecture.
    - Instead of passing the output of the LSTM layer through a linear layer, I have taken the cosine similarity between the output embeddings of the two sentences. The cosine similarity is then passed through a sigmoid function to get a probability between 0 to 1.
    - This approach yeiled worse results as compared to the linear layer approach. The accuracy dropped from 55.80% to 44.2% on the PAWS dataset.
2. Catastrophic Forgetting:
    - I have tried to unfreeze the pretrained embeddings and train the model on the PAWS dataset.
    - There was no improvement in the accuracy.

### Dataset Analysis

I have calculated the average lexical overlap between the sentence pairs in the PAWS dataset. The average lexical overlap was as follows:
- Train: 0.889
- Dev: 0.888
- Test: 0.890

The lexical overlap was calculated using the following formula:
```python
lexical_overlap = set(common_words)/set(words_in_sentence1 + words_in_sentence2)
```
The sentences were tokenized and preprocessed, and lexical overlap was calculated on the preprocessed sentences.

I have also plotted the distribution of labels (0/1) in the PAWS dataset. It seems to have around 55% of 0 labels and 45% of 1 labels.

## Phrase Similarity

In this task, I have the BILSTM model(similar to the one used in the sentence similarity task) with three different approaches to get the embeddings of the phrases. I have used pre-trained Glove embeddings for the words.


1. **Phrase Only** :
    - In this approach, I just passed the phrases through the BILSTM model to get the embeddings of the phrases, rest of the process is the same as the sentence similarity task.
2. **Phrase + Sentence** :
    - In this approach, I passed the phrases and the sentences through the BILSTM model to get the embeddings of the phrases and the sentences.
    - The embeddings of the phrases and the sentences are concatenated. So, the representation of the phrase is a combination of the representation of the phrase and the sentence, the rest of the process is same as the sentence similarity task.
3. **Phrase Attention** :
    - In  this approach, I passed the phrases and the sentences through the BILSTM model to get the embeddings of the phrases and the sentences.
    - Then, using the phrase and corresponding sentence embeddings, I calculate the context vector that encapsulates the relevant information from the sentence with respect to the phrase using attention mechanism.
    - The context vectors for both phrases are concatenated and passed through a linear layer, the rest of the process is the same as the sentence similarity task.


### Results

The results are as follows:
<!-- Table -->
| Model | PiC |
| :---: | :---: |
| Phrase Only | 32.22% |
| Phrase + Sentence | 47.15% |
| Phrase Attention | 50.00% |

As we can see, the accuracy is very bad for the Phrase Only model. This is because the phrase embeddings are not able to capture the context of the phrase in the sentence.

The Phrase + Sentence model performs better than the Phrase-only model because the phrase embeddings are a combination of the phrase and the sentence embeddings. Thus, the phrase embeddings can capture the context of the phrase to some extent.

The Phrase Attention model performs better than the Phrase + Sentence model because the context vector is calculated using the attention mechanism. Thus, the context vector can capture the relevant information from the sentence concerning the phrase.

### Dataset Analysis

The PiC dataset is an effective dataset to test the performance of models on phrase similarity i.e. compare the semantic similarity of two phrases in the same context sentence.

I have checked whether this assumption that 'the phrase is selected such that it's a substring of the sentence' is true or not. I have calculated the number of phrases that are not substrings of the sentence and found only 4 such cases where phrases were not substrings of the sentence out of 10k datapoints.

