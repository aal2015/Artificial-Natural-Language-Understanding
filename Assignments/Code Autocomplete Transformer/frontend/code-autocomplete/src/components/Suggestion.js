import Button from '@mui/material/Button';

function Suggestion(props) {

    return (<>
        <Button
            variant="contained"
            id="clear-suggestions-button"
            color="warning"
            onClick={props.onClearSuggestions}
        >
            Clear Suggestions
        </Button>
        <div className="component-padding suggestion-component">
            <div className="suggestion-box">
                {props.codeSuggestions.map((suggestion, id) => (

                    <p key={id}>{suggestion}</p>

                ))}
            </div>
        </div>
    </>)
}

export default Suggestion;