import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';

function LangInput(props) {
    return (
        <>
            <h2>Hindi</h2>
            <div className="text-button-spacing">
                <TextField
                    label="Enter Hindi Text"
                    value={props.value}
                    onChange={props.changeHandler}
                    fullWidth
                    multiline
                    rows={10}
                />
            </div>
            <Button
                variant="contained"
                onClick={props.onTranslate}
            >
                Translate!
            </Button>
        </>
    )
}

export default LangInput;