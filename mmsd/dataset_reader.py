from typing import Dict, Iterable, Optional

from allennlp.data import Instance, Tokenizer, TokenIndexer, Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from overrides import overrides


@DatasetReader.register('sarcasm')
class SarcasmDatasetReader(DatasetReader):
    def __init__(self, lazy: bool = False, tokenizer: Optional[Tokenizer] = None,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None, tiny_sample: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._tiny_sample = tiny_sample

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        pass

    @overrides
    def text_to_instance(self, text: str, is_sarcasm: Optional[bool]) -> Instance:
        fields = {'tokens': TextField(self._tokenizer.tokenize(text), self._token_indexers)}

        if is_sarcasm is not None:
            fields['label'] = LabelField(str(is_sarcasm))

        return Instance(fields)
