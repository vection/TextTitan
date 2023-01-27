from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import DebertaV2Model


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        classifier_dropout = 0.1
        self.dense = [nn.Linear(config.hidden_size, 128), nn.Tanh(), nn.Dropout(classifier_dropout), nn.Linear(128, 32),
                      nn.Tanh(), nn.Dropout(classifier_dropout),
                      nn.Linear(32, config.num_labels)]
        self.dense = nn.Sequential(*self.dense)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        #         x = torch.tanh(x)
        #         x = self.dropout(x)
        #         x = self.out_proj(x)
        return x



class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

# deberta_V3 model
class DebertaV3(nn.Module):
    def __init__(self, output_dim,base="microsoft/deberta-v3-base",fc_layer=768):
        super().__init__()
        self.deberta = DebertaV2Model.from_pretrained(base)
        # self.classifier = RobertaClassificationHead(config)
        self.average_pooling = MeanPooling()
        self.output_dim = output_dim
        self.num_neurons = fc_layer
        self.fc_hidden = 64
        self.hidden_size = 32
        self.fc = nn.Sequential(*[nn.Linear(self.num_neurons, self.fc_hidden), nn.ReLU(inplace=True),
                                  nn.Linear(self.fc_hidden, self.hidden_size), nn.ReLU(inplace=True),
                                  nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(inplace=True),
                                  nn.Linear(self.hidden_size, self.output_dim)])

        dropout = 0.1

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            spelling_mistakes=None,
            contractions=None,
            symbols=None,
            unique=None
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids, output_hidden_states=True)

        pooling = self.average_pooling(outputs.last_hidden_state, attention_mask)

        logits = self.fc(pooling)
        # logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = nn.SmoothL1Loss()
            logits = torch.squeeze(logits, -1)

            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )