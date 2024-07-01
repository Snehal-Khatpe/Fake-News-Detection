# Fake-News-Detection

# Summary
This project implements a machine learning model to detect fake news articles. 
It uses a dataset of labeled real and fake news, preprocesses the text data, builds a deep learning model with an embedding layer and bidirectional LSTM, 
and trains it to classify articles as real or fake. 
The model achieves high accuracy on the test set. 
The code demonstrates the full pipeline from data loading and cleaning to model training and evaluation.

Data Loading and Preprocessing
* Loads datasets of real and fake news articles from CSV files
* Combines real and fake datasets and adds labels
* Cleans text by removing stopwords and tokenizing
* Creates word clouds to visualize frequent words in real vs fake news
* Splits data into training and test sets
  
Text Processing
* Tokenizes text and converts to sequences of integers
* Pads sequences to uniform length for model input
* Uses a vocabulary of the most common words
  
Model Architecture
* Uses an embedding layer to create word vectors
* Employs a bidirectional LSTM layer for sequence processing
* Has dense layers for final classification
* Uses binary cross-entropy loss and Adam optimizer
  
Model Training
* Trains for 2 epochs with batch size of 64
* Uses 10% of training data for validation
* Achieves very high accuracy on training and validation sets

Model Evaluation
* Evaluates on held-out test set
* Achieves 99.75% accuracy on test data
* Generates confusion matrix to visualize results
  
Visualizations
* Word clouds of frequent terms in real vs fake news
* Distribution of article lengths
* Confusion matrix of model predictions
  
Results
1. The model achieved an impressive accuracy of 99.75% on the test set. This indicates that the model is highly effective at distinguishing between real and fake news articles.
2. The confusion matrix provides a detailed breakdown of the model's performance: [[4702,7], [15,4256]]
* True Negatives (correctly identified fake news): 4702
* False Positives (real news incorrectly classified as fake): 7
* False Negatives (fake news incorrectly classified as real): 15
* True Positives (correctly identified real news): 4256
3. Precision and Recall that we can infer:
    * Precision for fake news: 4702 / (4702 + 15) ≈ 99.68%
    * Recall for fake news: 4702 / (4702 + 7) ≈ 99.85%
    * Precision for real news: 4256 / (4256 + 7) ≈ 99.84%
    * Recall for real news: 4256 / (4256 + 15) ≈ 99.65%
These high values indicate that the model is performing exceptionally well in both identifying fake news and real news.

4. The model's training history shows:
    * Epoch 1: accuracy of 98.67% on training data, 99.86% on validation data
    * Epoch 2: accuracy of 99.98% on training data, 99.86% on validation data
The high accuracy achieved in just two epochs suggests that the model quickly learned to differentiate between real and fake news. 
The consistency between training and validation accuracy indicates that the model is not overfitting.

Downsides to consider:
1. The extremely high accuracy (99.75%) on the test set could potentially indicate some degree of overfitting to the specific characteristics of the dataset used. This might lead to decreased performance on significantly different data.
2. If the new data comes from sources that were not well-represented in the training data (e.g., different countries, niche publications, or new online platforms), the model's performance might decrease.
3. The "black box" nature of deep learning models makes it difficult to explain exactly why an article was classified as fake or real.

# Why  this architecture:
1. Sequential Model:
2. The project uses a sequential model, which is a linear stack of layers. This choice allows for a straightforward,
    layer-by-layer architecture that is well-suited for text classification tasks.
4. Embedding Layer:
    * Purpose: Converts words (represented as integers) into dense vectors of fixed size.
    * Rationale: Embeddings capture semantic relationships between words, allowing the model to understand word meanings and contexts.
      This is crucial for understanding the nuances in news articles.
5. Bidirectional LSTM (Long Short-Term Memory):
    * Structure: model.add(Bidirectional(LSTM(128)))
    * Purpose: Processes the sequence in both forward and backward directions.
    * Rationale:
    * LSTMs are excellent for capturing long-term dependencies in text, which is vital for understanding context in news articles.
    * The bidirectional aspect allows the model to understand context from both past and future words, which is particularly useful
      for detecting subtle indicators of fake news that might depend on the full context of a sentence or paragraph.
    * 128 units provide a balance between model complexity and computational efficiency.
6. Dense Layers:
    * First Dense Layer: model.add(Dense(128, activation = 'relu'))
    * Purpose: Adds non-linearity and learns high-level features.
    * Rationale: ReLU activation is used for its efficiency and ability to mitigate the vanishing gradient problem.
* Output Layer: model.add(Dense(1, activation= 'sigmoid'))
* Purpose: Produces the final classification (fake or real).
* Rationale: Sigmoid activation is used for binary classification, outputting a probability between 0 and 1.

