# Code Autocomplete
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
![starting page](https://user-images.githubusercontent.com/28766535/220980681-f28d755f-98c9-4011-811f-d32b8287741a.png)

<b>Suggesting Potention Code Completion after Python Code Input</b>
![Code Suggestion after Python Code Input](https://user-images.githubusercontent.com/28766535/220980908-faeca274-034e-48c5-a6d1-a7fd07b0445b.png)

## References
- https://stackoverflow.com/questions/62166362/how-to-tokenize-python-code-using-the-tokenize-module - Tokenizing Python code
- https://www.youtube.com/watch?v=7LNl2JlZKHA - Creating Flask - React Project
- https://dev.to/ondiek/sending-data-from-react-to-flask-apm - For sending data from React to Flask.
- https://stackoverflow.com/questions/61245215/saving-vocabulary-object-from-pytorchs-torchtext-library - Saving torch object
