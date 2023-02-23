# Code Autocomplete
## About Project
A web app was created for suggesting code completions primarily for Pytorch code. This was done so by training a model to do code generation on the given Python code. This is explored in 'auto complete Abhinav.ipynb' file. File, 'backup_test.ipynb' contains the code used in Flask, especially in 'autocomplete.py', for suggesting code to complete the current Python code. Currently, the model is not giving good suggestions for code completion suggesting that better training, model, and/or dataset processing need to be done to get good code completion suggestions.

## Technologies used for Web App
<b>Frontend</b>
- React

<b>Backend</b>
- Flask

## Data
'codeparrot-clean' dataset from CodeParrot organization was used to train the model. The dataset was meant to train models for code generation. Since the main emphasis is given to Pytorch code generation, all first 1000 repo suspected to contain Pytorch code were extracted for model training.

## Note
It is important to run Flask server on port 5000 due to the proxy setting in package.json in React.

## References
- https://stackoverflow.com/questions/62166362/how-to-tokenize-python-code-using-the-tokenize-module - Tokenizing Python code
- https://www.youtube.com/watch?v=7LNl2JlZKHA - Creating Flask - React Project
- https://dev.to/ondiek/sending-data-from-react-to-flask-apm - For sending data from React to Flask.
- https://stackoverflow.com/questions/61245215/saving-vocabulary-object-from-pytorchs-torchtext-library - Saving torch object
