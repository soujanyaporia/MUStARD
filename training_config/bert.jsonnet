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
  train_data_path: 'data/sarcasm_data.json',
  model: {
    type: 'bert_for_classification',
    bert_model: pretrained_model
  },
  iterator: {
    type: 'bucket',
    sorting_keys: [['tokens', 'num_tokens']],
    batch_size: 32,
  },
  trainer: {
    num_epochs: 4,
    validation_metric: '+accuracy',
    optimizer: 'adam',
    learning_rate_scheduler: 'slanted_triangular'
  }
}
