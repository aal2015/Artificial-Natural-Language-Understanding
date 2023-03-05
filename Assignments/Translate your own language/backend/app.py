from flask import Flask, request, render_template, redirect, session
from translate import translate

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sentimentAnalysisKey'

@app.route('/hindiToEnglishTranslate', methods=["GET", "POST"])
def hindiToEnglishTranslation():
    hindi_text = request.json['hindi_text']
    tranlated_en_text = translate(hindi_text)
    return {"english_text": tranlated_en_text}

if __name__ == "__main__":
    app.run(debug=True)