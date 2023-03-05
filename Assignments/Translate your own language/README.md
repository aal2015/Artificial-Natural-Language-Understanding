# Hindi-English Translator

## About Project
A model was trained to translate Hindi text to English text. 'IITB-English-Hindi Parallel Corpus' was the dataset used for training the model. 
More information about the data can be found at https://huggingface.co/datasets/cfilt/iitb-english-hindi. The model training is done in 'Translate your own language Abhinav.ipynb' file. 'backend_test.ipynb' file illustrates the logic to translate from Hindi text to English text during inference for adding it in the backend.

## Tech Stack
- Frontend - Reach
- Backend - Flask

## Requiment
#### Required framework for running frontend
- Node js

#### Required packages for running backend
- stanza
- torch
- torchtext
- flask

## How to run
#### Running Flask
Note: Please uncomment 'stanza.download('hi')' in line 189 in the translate.py file in the backend when running for the first time. This will download necessary a model for Hindi text for tokenization.
- cd backend/
- python3 app.py

#### Running React
- cd frontend/hindi-english-translator/
- npm start

## Demo
### Home Page
![homepage](https://user-images.githubusercontent.com/28766535/222973398-429a4d0d-61af-4b1a-90d0-d19512aaace1.png)

### Translation from Hindi text to English text
![result](https://user-images.githubusercontent.com/28766535/222973464-ba85d874-f9cd-4acb-ac09-8e0cd65080bb.png)

Intended Translation: How are you?
