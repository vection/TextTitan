import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

from transformers.modeling_outputs import SequenceClassifierOutput


class RNN_LSTM(nn.Module):
    losses = {'cross_entropy': nn.CrossEntropyLoss(),
              'huber': nn.HuberLoss(),
              'mse': nn.MSELoss(),
              'smooth': nn.SmoothL1Loss()}

    def __init__(self, input_dim, embedding_dim, bidirectional, hidden_dim, output_dim, dropout, pad_idx,
                 num_layers=2, fc_hidden1=128, fc_hidden2=64, loss_func='cross_entropy'):
        super().__init__()
        self.lstm_hidden = hidden_dim
        self.fc_hidden2 = fc_hidden2
        self.fc_hidden = fc_hidden1
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.pad_idx = pad_idx
        self.loss_func_name = loss_func
        self.loss_func = self.losses[loss_func]
        self.label_map = None

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)

        # embedded to cuda if available
        self.rnn = nn.LSTM(embedding_dim,
                           self.lstm_hidden,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.num_neurons = (self.lstm_hidden * num_layers)

        self.fc = [nn.Linear(self.num_neurons, self.fc_hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                   nn.Linear(self.fc_hidden, self.fc_hidden2), nn.ReLU(inplace=True),
                   nn.Dropout(dropout), nn.Linear(self.fc_hidden2, output_dim)]
        self.fc = nn.Sequential(*self.fc)
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, **kwargs):
        embedded = self.dropout(self.embedding(input_ids))
        packed_output, (hidden, cell) = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        hidden = self.fc(hidden)
        loss = None
        if 'labels' in kwargs:
            loss = self.loss_func(hidden, kwargs['labels'])
        return SequenceClassifierOutput(loss=loss, logits=hidden)

    def save_pretrained(self, path):
        if '.pth' not in path:
            path += '.pth'

        conf = self.state_dict()
        conf['lstm_hidden'] = self.lstm_hidden
        conf['fc_hidden1'] = self.fc_hidden
        conf['fc_hidden2'] = self.fc_hidden2
        conf['input_dim'] = self.input_dim
        conf['embedding_dim'] = self.embedding_dim
        conf['bidirectional'] = self.bidirectional
        conf['num_layers'] = self.num_layers
        conf['output_dim'] = self.output_dim
        conf['dropout'] = self.dropout_rate
        conf['pad_idx'] = self.pad_idx
        conf['loss_func_name'] = self.loss_func_name
        conf['label_map'] = self.label_map
        conf['model_name'] = 'rnn-lstm'
        torch.save(conf, path)
        print("Model Saved as ", path)

    @staticmethod
    def load(path):
        conf = torch.load(path)

        model = RNN_LSTM(conf['input_dim'], conf['embedding_dim'], conf['bidirectional'], conf['lstm_hidden'],
                         conf['num_layers'], conf['output_dim'], conf['dropout'], conf['pad_idx'],
                         conf['fc_hidden1'], conf['fc_hidden2'], conf['loss_func_name'])
        model.label_map = conf['label_map']

        keys_to_drop = ['lstm_hidden', 'fc_hidden1', 'fc_hidden2', 'input_dim', 'embedding_dim', 'bidirectional',
                        'num_layers', 'output_dim',
                        'dropout', 'pad_idx', 'loss_func_name', 'label_map', 'model_name']
        for key in keys_to_drop:
            conf.pop(key)

        model.load_state_dict(conf)

        return model


