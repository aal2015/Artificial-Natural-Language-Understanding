import styles from './DisplayTranslation.module.css';
import Button from '@mui/material/Button';

function DisplayTranslation(props) {
    return (
        <>
            <h2>English</h2>
            <div
                id={styles['display-background']}
                className="text-button-spacing"
            >
                {props.translatedText}
            </div>
            <Button
                variant="contained"
                onClick={props.onClearTranslation}
            >
                Clear
            </Button>
        </>
    )
}

export default DisplayTranslation;