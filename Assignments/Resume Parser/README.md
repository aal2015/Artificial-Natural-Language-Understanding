# Resume Parser
## How to run
- pip install -r requirements.txt
- python3 app.py 
Running inside vitual environment (https://github.com/jakerieger/FlaskIntroduction)
- pip install virtualenv
- virtualenv env
- source env/bin/activate
- pip install -r requirements.txt
- python app.py

## About Project
A web application using Flask was used to create a resume parser. Currently, it parses
only for skills and education. The Main logic for parsing is inside the file 'resume_parser.py'.

## Working of the Project
To extract desired credentials, spaCy was used. From spaCy, a pipeline component, 'entity ruler', was added. It is a rule-based system from entity
recognition from the defined token-based rules or exact phrase matches. In the case of the project, data was collected for exact phrase matches
from which information would be extracted. 

## Data for Entity Ruler
Data related to skills and education was collected. For the skills dataset, a dataset thatwas used during the lab for NLU class was used in this project. 
For education data, data from this link, https://thebestschools.org/degrees/college-degree-levels/ was taken. Data from Ati Chetsurakul hosting in the 
GitHub link, https://github.com/AtiChetsurakul/NLP_labsession/tree/main/Hw3m was also used.

## References
Flask tutorial
- https://www.youtube.com/watch?v=Z1RJmh_OqeA&t=1670s

Uploading files in Flask app
- https://www.youtube.com/watch?v=GeiUTkSAJPs&t=463s
- https://flask.palletsprojects.com/en/2.2.x/patterns/fileuploads/


# Demo of the Running Application
#### Home Page
![Home Page](https://user-images.githubusercontent.com/28766535/217636557-ca806336-5af8-4baa-85e3-1f8945b09a94.png)

#### Uploading Resume
![Uploading Resume ](https://user-images.githubusercontent.com/28766535/217636892-c518c628-0cee-4eac-b825-3be33f819658.png)

#### Result Page of Parsed Resume 
![Result Page after parsing Resume](https://user-images.githubusercontent.com/28766535/217636972-a1afda82-8f41-4d58-b33c-769fe1723769.png)
