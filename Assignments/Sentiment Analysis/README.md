# Sentiiment Analysis
## How to run
- pip install flask praw matplotlib
- python app.py
- pip install -U spacy
- python -m spacy download en_core_web_md

## About Project
A web application using Flask was used to do sentiment analysis on Reddit posts. Originally, the intend was to do analysis on Twitter posts but due to permission issues, Reddit was selected instead.  To do sentiment analysis, a module was trained using 'Stanford Sentiment Treebank'. It is explored in 'Sentiment Analysis Abhinav.ipynb'. In that file, double negation and negative positive sentence test was also performed to test the model. 'reddit_sentiment_analysis.ipynb' file shows the sample working of the model on doing sentiment analysis on Reddit posts on a certain Subreddit.

# Demo of the Running Application
#### Home Page
![homepage](https://user-images.githubusercontent.com/28766535/219431802-e1b71180-4054-4334-966e-5caf60100152.png)

#### Sentiment Analysis Page
![result page 1](https://user-images.githubusercontent.com/28766535/219431866-e85e6c4d-2985-42f5-a9ac-0dd48ffc6d55.png)
![result page 2](https://user-images.githubusercontent.com/28766535/219431883-f69beb74-cfe4-4017-9bbf-a47d22c2943f.png)
![result page 3](https://user-images.githubusercontent.com/28766535/219431893-6a5c2180-0d54-434b-9a4d-7dfd61f95135.png)
