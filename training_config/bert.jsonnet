local pretrained_model = 'bert-base-cased';

{
  dataset_reader: {
    type: 'sarcasm',
    tokenizer: {
      type: 'word',
      word_splitter: {
        type: 'bert-basic',
        do_lower_case: false
      }
    },
    token_indexers: {
      bert: {
        type: 'bert-pretrained',
        pretrained_model: pretrained_model,
        do_lowercase: false,
      }
    },
    //tiny_sample: true
  },
  train_data_path: 'data/sarcasm_speaker_independent_train.jsonnet',
  validation_data_path: 'data/sarcasm_speaker_independent_test.jsonnet',
  model: {
    type: 'bert_for_classification_plus',
    bert_model: pretrained_model
  },
  iterator: {
    type: 'bucket',
    sorting_keys: [['tokens', 'num_tokens']],
    batch_size: 32,
  },
  trainer: {
    num_epochs: 40,
    patience: 10,
    validation_metric: '+f1',
    optimizer: 'ranger',
//    learning_rate_scheduler: {
//      type: 'slanted_triangular',
//      num_epochs: 4,
//      num_steps_per_epoch: 10,
//    }
  }
}
