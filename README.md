# MMSD: Multimodal Sarcasm Detection

This repository contains the dataset and code for our ACL 2019 paper "Towards Multimodal Sarcasm Detection
(An _Obviously_ Perfect Paper)".

We release the MMSD dataset which is a multimodal video corpus for research in automated sarcasm discovery. The dataset
is compiled from popular TV shows including *Friends*, *The Golden Girls*, *The Big Bang Theory*, and
*Sarcasmaholics Anonymous*. MMSD consists of audiovisual utterances annotated with sarcasm labels. Each utterance is
accompanied by its context, which provides additional information on the scenario where the utterance occurs.

## Example Instance

![Example instance](images/utterance_example.jpg)

<p align="center"> Example sarcastic utterance from the dataset along with its context and transcript. </p>     

## Raw Videos

We provide a [Google Drive folder with the raw video clips](https://drive.google.com/file/d/1i9ixalVcXskA5_BkNnbR60sqJqvGyi6E/view?usp=sharing),
including both the utterances and their respective context

## Data Format

The annotations and transcripts of the audiovisual clips are available at [`data/sarcasm_data.json`](data/sarcasm_data.json).
Each instance in the JSON file is allotted one identifier (e.g. "1\_60") which is a dictionary of the following items:   


| Key                     | Value                                                                          | 
| ----------------------- |:------------------------------------------------------------------------------:| 
| `utterance`             | The text of the target utterance to classify.                                  | 
| `speaker`               | Speaker of the target utterance.                                               | 
| `context`               | List of utterances (in chronological order) preceding the target utterance.    | 
| `context_speakers`      | Respective speakers of the context utterances.                                 | 
| `sarcasm`               | Binary label for sarcasm tag.                                                         | 

Example format in JSON:

```json
{
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
}
```

## Citation

Please cite the following paper if you find this dataset useful in your research:

```tex
@inproceedings{mmsd,
    title = "Towards Multimodal Sarcasm Detection (An  _Obviously_ Perfect Paper)",
    author = "Castro, Santiago  and
      Hazarika, Devamanyu  and
      P{\'e}rez-Rosas, Ver{\'o}nica  and
      Zimmermann, Roger  and
      Mihalcea, Rada  and
      Poria, Soujanya",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = "jul",
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
}
```

## Run the code

1. Setup an environment with Conda:

    ```bash
    conda env create -f environment.yml
    conda activate mmsd
    python -c "import nltk; nltk.download('punkt')"
    ```

2. Download [Common Crawl pretrained GloVe word vectors of size 300d, 840B tokens](http://nlp.stanford.edu/data/glove.840B.300d.zip)
somewhere.

3. [Extract the visual features](visual/README.md).

4. Extract BERT features in another environment with Python 2 and TensorFlow 1.11.0 following
["Using BERT to extract fixed feature vectors (like ELMo)" from BERT's repo](https://github.com/google-research/bert/tree/d66a146741588fb208450bde15aa7db143baaa69#using-bert-to-extract-fixed-feature-vectors-like-elmo)
and running this command:

    ```bash
    # Download BERT-base uncased in some dir:
    wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
    # Then put the location in this var:
    BERT_BASE_DIR=...
    
    python extract_features.py \
      --input_file=data/bert-input.txt \
      --output_file=data/bert-output.jsonl \
      --vocab_file=${BERT_BASE_DIR}/vocab.txt \
      --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
      --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
      --layers=-1,-2,-3,-4 \
      --max_seq_length=128 \
      --batch_size=8
    ```   
    
    To download existing BERT features for the utterances, use [this link](https://drive.google.com/file/d/1GYv74vN80iX_IkEmkJhkjDRGxLvraWuZ/view?usp=sharing) to download two files which should be placed as `data/bert-output.jsonl` and `data/bert-output-context.jsonl`

5. Modify [`config.py`](config.py) and run:

    ```bash
    python train_svm.py
    ```

6. Evaluation: We evaluate using weighted F-score metric in a 5-fold cross validation scheme. The fold indices are available at `data/split_incides.p` . Refer to our baseline scripts for more details.
