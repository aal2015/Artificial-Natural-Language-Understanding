# Code Autocomplete
## About Project
A web app was created for suggesting code completion primarily for Pytorch code.

## Technologies used for Web App
<b>Frontend</b>
- React

<b>Backend</b>
- Flask

## Data
'codeparrot-clean' dataset from CodeParrot organization was used to train the model. The dataset was meant to train models for code generation. Since the main emphasis is given to Pytorch code generation, all first 1000 repo suspected to contain Pytorch code were extracted for model training.

## Note
It is important to run Flask server on port 5000 due to proxy setting in package.json in React.

## References
- https://stackoverflow.com/questions/62166362/how-to-tokenize-python-code-using-the-tokenize-module - Tokenizing Python code
- https://www.youtube.com/watch?v=7LNl2JlZKHA - Creating Flask - React Project
- https://dev.to/ondiek/sending-data-from-react-to-flask-apm - For sending data from React to Flask.
