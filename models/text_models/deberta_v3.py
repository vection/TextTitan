from typing import Optional
import torch
import torch.nn as nn
from transformers import DebertaV2Model
from transformers.modeling_outputs import SequenceClassifierOutput


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, fc_hidden_1=64, fc_hidden_2=32, classifier_dropout=0.1):
        super().__init__()
        self.classifier_dropout = classifier_dropout
        self.dense = [nn.Linear(config.hidden_size, fc_hidden_1), nn.Tanh(), nn.Dropout(classifier_dropout),
                      nn.Linear(fc_hidden_1, fc_hidden_2), nn.Tanh(), nn.Dropout(classifier_dropout),
                      nn.Linear(fc_hidden_2, config.num_labels)]
        self.dense = nn.Sequential(*self.dense)
        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
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


class DebertaV3(nn.Module):
    def __init__(self, base_image="microsoft/deberta-v3-base", problem_type='single_label_classification',
                 fc_layer_1=64, fc_layer_2=32, output_dim=1, dp=0.1):
        super().__init__()
        self.deberta = DebertaV2Model.from_pretrained(base_image)  # RobertaModel(config, add_pooling_layer=False)
        # self.classifier = RobertaClassificationHead(config)
        self.problem_type = problem_type
        self.average_pooling = MeanPooling()
        self.config = self.deberta.config
        self.num_labels = output_dim
        self.config.num_labels = output_dim
        self.config.problem_type = problem_type
        self.classifier_head = ClassificationHead(self.config, fc_layer_1, fc_layer_2, dp)
        # self.base_num_neurons = self.config.hidden_state
        self.fc_hidden_1 = fc_layer_1
        self.fc_hidden_2 = fc_layer_2
        self.output_dim = output_dim
        self.dropout = dp
        self.base_image = base_image
        self.label_map = None

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

        logits = self.classifier_head(pooling)
        loss = None
        if labels is not None:
            if 'regression' in self.config.problem_type:
                if self.config.problem_type == "regression_mse":
                    loss_fct = nn.MSELoss()
                elif self.config.problem_type == "regression_hubber":
                    loss_fct = nn.HuberLoss()
                elif self.config.problem_type == "regression_smooth":
                    loss_fct = nn.SmoothL1Loss()

                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.float().squeeze())
                else:
                    loss = loss_fct(logits, labels)

            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def save_pretrained(self, path):
        if '.pth' not in path:
            path += '.pth'

        conf = self.state_dict()
        conf['problem_type'] = self.problem_type
        conf['fc_hidden1'] = self.fc_hidden_1
        conf['fc_hidden2'] = self.fc_hidden_2
        conf['output_dim'] = self.output_dim
        conf['base_image'] = self.base_image
        conf['label_map'] = self.label_map
        conf['dropout'] = self.dropout
        conf['model_name'] = 'deberta'
        torch.save(conf, path)
        print("Model Saved as ", path)

    @staticmethod
    def load(path):
        conf = torch.load(path)

        model = DebertaV3(problem_type=conf['problem_type'], fc_layer_1=conf['fc_hidden1'],
                          fc_layer_2=conf['fc_hidden2'], output_dim=conf['output_dim'], dp=conf['dropout'])
        model.label_map = conf['label_map']

        keys_to_drop = ['problem_type', 'fc_hidden1', 'fc_hidden2', 'output_dim', 'dropout', 'label_map', 'base_image',
                        'model_name']
        for key in keys_to_drop:
            conf.pop(key)

        model.load_state_dict(conf)

        return model

