# Code Autocomplete Transformer
## About Project
This project is based on the 'Code Autocomplete' Project, https://github.com/aal2015/Natural-Language-Understanding/tree/main/Assignments/Code%20Autocomplete. Instead of using LSTM, the decoder part of the transformer was used.. This is explored in 'Improved Language Modeling - Abhinav.ipynb' file. File, 'backup_test.ipynb' contains the code used in Flask, especially in 'autocomplete.py', for suggesting code to complete the current Python code. Currently, the transformer model is performing worse then LSTM model suggesting that more work needs to be done to get good code suggestions.

## Technologies used for Web App
<b>Frontend</b>
- React

<b>Backend</b>
- Flask

## Data
'codeparrot-clean' dataset from CodeParrot organization was used to train the model. The dataset was meant to train models for code generation. Since the main emphasis is given to Pytorch code generation, all first 1000 repo suspected to contain Pytorch code were extracted for model training.

## Note
It is important to run Flask server on port 5000 due to the proxy setting in package.json in React. Also all dependencies for running React app need to be installed.

## Demo
<b>Starting Page</b>
![start](https://user-images.githubusercontent.com/28766535/224557978-133959d7-5c87-434e-8c6b-b4f6ce20596b.png)
<b>Suggesting Potention Code Completion after Python Code Input</b>
![end](https://user-images.githubusercontent.com/28766535/224557995-e5fce01d-8351-40be-94ac-2f51f2868d93.png)

## Limitations
- Suggests only one code per input

## Future Work
- Compare Greedy Decoding vs. Beam Search
- Use Beam Search to diversify code suggestiosn.

## References
- https://stackoverflow.com/questions/62166362/how-to-tokenize-python-code-using-the-tokenize-module - Tokenizing Python code
- https://www.youtube.com/watch?v=7LNl2JlZKHA - Creating Flask - React Project
- https://dev.to/ondiek/sending-data-from-react-to-flask-apm - For sending data from React to Flask.
- https://stackoverflow.com/questions/61245215/saving-vocabulary-object-from-pytorchs-torchtext-library - Saving torch object
