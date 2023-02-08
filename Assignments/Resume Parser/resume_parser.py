# Imports
import pandas as pd
# import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from PyPDF2 import PdfReader

nlp = spacy.load('en_core_web_md')
skill_path = "static/pattern_data/education_skill.jsonl"

ruler = nlp.add_pipe("entity_ruler")
ruler.from_disk(skill_path)

# Function for cleaning resume data
### Remove all stop words, punctuations, spaces, and symbols
### lowercase all tokens
def preprocessing(sentence):
    
    stopwords = list(STOP_WORDS)
    doc = nlp(sentence)
    cleaned_tokens = []
    
    for token in doc:
        if token.text not in stopwords and token.pos_ != 'PUNCT' and token.pos_ != 'SPACE' and \
            token.pos_ != 'SYM':
                cleaned_tokens.append(token.lemma_.lower().strip())
                
    return " ".join(cleaned_tokens)

# Function for listing unique skills
def unique_skills_and_education(doc):
    skills    = []
    education = []

    for ent in doc.ents:
        if ent.label_ == 'SKILL':
            skills.append(ent.text)
        elif ent.label_ == 'EDUCATION':
            education.append(ent.text)
    
    # Convert from set to list type
    education_list = list(set(education))
    # Reverse list to dispaly education in order
    education_list.reverse()
    return set(skills), education_list

# Main function to start parsing resume <---------------------
def extract_skills_education(filePath):
    reader = PdfReader(filePath)
    number_of_pages = len(reader.pages)

    # Extract text from uploaded resume from all pages
    text = ''
    for page_number in range(number_of_pages):
        page = reader.pages[page_number]
        text += ' ' + page.extract_text() 
    
    # Clean text
    text = preprocessing(text)

    # Tokenize text
    doc = nlp(text)

    # Get unique skills and eduation
    skills, education = unique_skills_and_education(doc)

    return skills, education