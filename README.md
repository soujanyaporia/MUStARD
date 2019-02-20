# MMSD: Multimodal Sarcasm Detection

## Data Format

Sample instance from the dataset. Each instance is alloted one identifier (e.g. 1_60) which comprises a dictionary of the following items:   


| Key                     | Value                                                                            | 
| ----------------------- |:------------------------------------------------------------------------------:  | 
| `utterance`             | The text of the target utterance to classify.                                    | 
| `speaker`               | Speaker of the target utterance.                                                 | 
| `context`               | List of utterances (in chronological order) preceding the target utterance.     | 
| `context_speakers`      | Respective speakers of the context utterances.                                   | 
| `sarcasm`               | Label for sarcasm tag.                                                          | 

Sample format in json:   

```
"1_60": {
        "utterance": "It's just a privilege to watch your mind at work.",
        "speaker": "SHELDON",
        "context": [
            "I never would have identified the fingerprints of string theory in the aftermath of the Big Bang.",
            "My apologies. What's your plan?"
        ],
        "context_speakers": [
            "LEONARD",
            "SHELDON"
        ],
        "sarcasm": true
    }
```
