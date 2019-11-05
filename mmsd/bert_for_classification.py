from typing import Dict, Optional, Union

import torch
from allennlp.data import Vocabulary
from allennlp.models import BertForClassification, Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import F1Measure
from overrides import overrides
from pytorch_pretrained_bert import BertModel


@Model.register("bert_for_classification_plus")
class BertForClassificationPlus(BertForClassification):
    def __init__(self, vocab: Vocabulary, bert_model: Union[str, BertModel], dropout: float = 0.0,
                 num_labels: int = None, index: str = "bert", label_namespace: str = "labels", trainable: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,) -> None:
        super().__init__(vocab, bert_model, dropout, num_labels, index, label_namespace, trainable, initializer,
                         regularizer)
        self._f1_measure = F1Measure(vocab.get_token_index(str(True), namespace=self._label_namespace))

    @overrides
    def forward(self, tokens: Dict[str, torch.LongTensor], label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        output_dict = super().forward(tokens, label)

        if label is not None:
            self._f1_measure(output_dict['logits'], label)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = super().get_metrics(reset)
        metrics['precision'], metrics['recall'], metrics['f1'] = self._f1_measure.get_metric(reset)
        return metrics
