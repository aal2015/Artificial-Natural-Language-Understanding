from flask import Flask, request, render_template, redirect, session
from autocomplete import get_code_suggestion

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sentimentAnalysisKey'

@app.route('/codeSuggestions', methods=["GET", "POST"])
def fetch_suggestions():
    prompt = request.json['prompt']
    print(prompt)
    suggestions = get_code_suggestion(prompt)

    return {'suggestions': suggestions}

if __name__ == "__main__":
    app.run(debug=True)