# Sentiment-Analysis-Using-NLP
Sentiment Analysis on Customer Reviews Using NLP

## Introduction
  Sentiment analysis, a subset of Natural Language Processing (NLP), aims to automatically determine the sentiment expressed in textual data, classifying it as positive, negative, or neutral. This project focuses    on **Sentiment Analysis on Customer Reviews Using NLP**, leveraging deep learning techniques to classify reviews and provide meaningful insights into customer opinions.

  The primary dataset utilized for this project is the **IMDB Movie Reviews Dataset**, a large collection of labeled movie reviews. The project begins with a baseline analysis using **Logistic Regression**,          followed by the actual implementation of a **Long Short-Term Memory (LSTM) network**, an advanced deep learning model designed to handle sequential text data effectively. Additionally, a **Flask based Web application** is developed to provide an interactive interface where users can input reviews and receive sentiment predictions in real time.

## Approach
  ## 1.Data collection
    The IMDB Movie Reviews Dataset is used here for sentiment analysis, containing 50,000 reviews labeled as positive or negative. 
    The dataset is typically split into training and testing subsets in an 80:20 ratio, ensuring a balanced approach for learning 
    and evaluation. It serves as a reliable benchmark for sentiment classification tasks in NLP.
  ## 2.Data Preprocessing
    Data preprocessing involves several steps to prepare the text for sentiment analysis. 
    * Text normalization ensures uniformity by converting all text to lowercase and removing unnecessary characters such as numbers 
      and HTML tags. 
    * Stopword removal eliminates common words like "is," "the," and "and" that do not contribute to sentiment analysis. 
    * Punctuation and special character removal helps clean the text by eliminating symbols and unnecessary marks. 
    * Tokenization breaks down text into individual words or tokens for further processing, while lemmatization reduces words to 
      their base form (e.g., "running" → "run") to standardize input. 
    Additionally, if the dataset is imbalanced, techniques like oversampling or undersampling may be applied to ensure a balanced 
    distribution of classes.
  ## 3.Feature Extraction
    Feature extraction transforms text into numerical representations for model training. 
    * For traditional machine learning models, TF-IDF Vectorization is used to convert text into numerical vectors. 
    * For deep learning models, word embeddings like Word2Vec and GloVe capture semantic relationships between words, 
      enhancing contextual understanding. 
    Additionally, tokenization and padding using TensorFlow/Keras convert text sequences into fixed-size numerical arrays, 
    making them suitable for LSTM input.
  ## 4.Model Building
    Model building involves training both a baseline and a deep learning model for sentiment classification. 
    * The baseline model, a Logistic Regression classifier, is trained using TF-IDF features to establish an initial performance 
      benchmark. While simple, it provides a useful reference for accuracy. 
    * For a more advanced approach, a deep learning model using LSTM is implemented. The architecture includes an embedding layer
      to convert words into dense vector representations, an LSTM layer to capture long-term dependencies in text, and dense layers
      for feature refinement. The output layer, equipped with a sigmoid activation function, performs binary classification. 
      The model is compiled with Binary Cross-Entropy Loss and optimized using Adam Optimizer, with training conducted via 
      Batch Gradient Descent to ensure optimal performance and proper validation.
  ## 5.Model Training and Evaluation
    Model training and evaluation involve training the LSTM model on the preprocessed dataset and assessing its performance 
    using various metrics. Accuracy is measured as the proportion of correct predictions, while precision, recall, and F1-score
    provide deeper insights into classification effectiveness. Additionally, a confusion matrix and AUC-ROC curve help visualize 
    and evaluate the model’s performance. To enhance accuracy and prevent overfitting, hyperparameter tuning is performed by 
    adjusting factors such as learning rate, dropout rate, and batch size
  ## 6.Web App Integration
    Web application integration involves developing a Flask-based web application to provide an intuitive and user-friendly 
    interface for sentiment analysis. Users can enter movie reviews through a text input field and click a predict button to 
    trigger the sentiment analysis model, which then displays real-time classification results as positive or negative. 
    The UI is designed using HTML, CSS, and JavaScript, ensuring a seamless and interactive experience.
## Results
  **Home Page**
  
  ![Screenshot (1)](https://github.com/user-attachments/assets/6672321c-6189-44e6-a236-a999c6ce20a6)

  **Page when review is positive**
  
  ![Screenshot (23)](https://github.com/user-attachments/assets/cb719f5f-24fd-41a4-97fe-6ce66fc3ddef)

  **Page when review is negative**
  
  ![Screenshot (24)](https://github.com/user-attachments/assets/c3ead7f9-4901-41f4-b580-cb7958b00780)



