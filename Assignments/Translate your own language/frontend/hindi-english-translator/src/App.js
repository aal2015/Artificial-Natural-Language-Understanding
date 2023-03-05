import { useState } from 'react';
import Grid from '@mui/material/Grid';
import LangInput from './components/LangInput';
import DisplayTranslation from './components/DisplayTranslation';
import './App.css';

function App() {
  const [langInput, setLangInput] = useState('');
  const [translated_en, setTranslated_en] = useState('');

  const langInputChangeHandler = event => {
    setLangInput(event.target.value);
  }

  const clearTranslation = () => {
    setTranslated_en('');
  }

  const translate = () => {
    console.log("Request Sent!!!");
    fetch('/hindiToEnglishTranslate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        hindi_text: langInput
      })
    }).then(
      res => res.json()
    ).then(
      res => {
        setTranslated_en(res.english_text);
        console.log("Recived Responses!!!");
      }
    );
  }

  return (
    <div className="App">
      <h1>Hindi-English Translator</h1>

      <Grid container>
        <Grid item xs={6} className="component-background">
          <LangInput
            value={langInput}
            changeHandler={langInputChangeHandler}
            onTranslate={translate}
          />
        </Grid>
        <Grid item xs={6} className="component-background">
          <DisplayTranslation
            translatedText={translated_en}
            onClearTranslation={clearTranslation}
          />
        </Grid>
      </Grid>
    </div>
  );
}

export default App;
