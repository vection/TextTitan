import torch.nn as nn
import torch
from transformers.modeling_outputs import SequenceClassifierOutput


class RNN_LSTM_CNN(nn.Module):
    losses = {'cross_entropy': nn.CrossEntropyLoss(),
              'huber': nn.HuberLoss(),
              'mse': nn.MSELoss(),
              'smooth': nn.SmoothL1Loss()}

    def __init__(self, input_dim, embedding_dim, bidirectional, hidden_dim, output_dim, dropout, pad_idx,
                 loss_func="cross_entropy",
                 fc_hidden2=32, cnn_layer1=100, cnn_layer2=20, fc_hidden1=64, k1_size=5, s1_size=3, k2_size=3,
                 s2_size=3, cnn_dp=0.5):

        super().__init__()
        self.lstm_hidden = hidden_dim
        self.fc_hidden2 = fc_hidden2
        self.fc_hidden1 = fc_hidden1
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional
        self.num_layers = 2
        self.output_dim = output_dim
        self.dropout = dropout
        self.pad_idx = pad_idx

        self.cnn_layer_1 = cnn_layer1
        self.cnn_layer_2 = cnn_layer2
        self.k1_size, self.s1_size, self.k2_size, self.s2_size = k1_size, s1_size, k2_size, s2_size
        self.cnn_dp = cnn_dp
        self.loss_func_name = loss_func
        self.loss_func = self.losses[loss_func]
        self.dropout_rate = dropout

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)

        # embedded to cuda if available
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=self.num_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.cnn1 = [nn.Conv1d(1, self.cnn_layer_1, kernel_size=k1_size, padding=1, stride=s1_size, bias=False),
                     nn.BatchNorm1d(self.cnn_layer_1), nn.Dropout(cnn_dp),
                     nn.ReLU(inplace=True)]
        self.cnn1 = nn.Sequential(*self.cnn1)

        self.cnn2 = [
            nn.Conv1d(self.cnn_layer_1, self.cnn_layer_2, kernel_size=k2_size, padding=1, stride=s2_size, bias=False),
            nn.BatchNorm1d(self.cnn_layer_2),
            nn.Dropout(cnn_dp), nn.ReLU(inplace=True)]
        self.cnn2 = nn.Sequential(*self.cnn2)

        sizes = self.cnn2(self.cnn1(torch.zeros([1, 1, hidden_dim * self.num_layers]))).size()
        self.num_neurons = (hidden_dim * self.num_layers) + sizes[1] * sizes[2]

        self.fc = [nn.Linear(self.num_neurons, self.fc_hidden1), nn.ReLU(inplace=True), nn.Dropout(dropout),
                   nn.Linear(self.fc_hidden1, self.fc_hidden2), nn.ReLU(inplace=True),
                   nn.Dropout(dropout), nn.Linear(self.fc_hidden2, output_dim)]
        self.fc = nn.Sequential(*self.fc)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, **kwargs):

        embedded = self.dropout(self.embedding(input_ids))
        packed_output, (hidden, cell) = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        x = hidden.unsqueeze(0).transpose(0, 1)
        output = self.cnn1(x)
        output = self.cnn2(output)
        output = output.view(output.shape[0], output.shape[1] * output.shape[2])
        res = torch.cat((output, hidden), dim=1)
        hidden = self.fc(res)

        loss = None
        if 'labels' in kwargs:
            loss = self.loss_func(hidden, kwargs['labels'])
        return SequenceClassifierOutput(loss=loss, logits=hidden)

    def save_pretrained(self, path):
        if '.pth' not in path:
            path += '.pth'
        conf = self.state_dict()
        conf['lstm_hidden'] = self.lstm_hidden
        conf['fc_hidden1'] = self.fc_hidden1
        conf['fc_hidden2'] = self.fc_hidden2
        conf['input_dim'] = self.input_dim
        conf['embedding_dim'] = self.embedding_dim
        conf['bidirectional'] = self.bidirectional
        conf['num_layers'] = self.num_layers
        conf['output_dim'] = self.output_dim
        conf['dropout'] = self.dropout_rate
        conf['pad_idx'] = self.pad_idx
        conf['cnn_layer_1'] = self.cnn_layer_1
        conf['cnn_layer_2'] = self.cnn_layer_2
        conf['cnn_dp'] = self.cnn_dp
        conf['k1_size'] = self.k1_size
        conf['s1_size'] = self.s1_size
        conf['k2_size'] = self.k2_size
        conf['s2_size'] = self.s2_size
        conf['cnn_dropout'] = self.cnn_dp
        conf['loss_func_name'] = self.loss_func_name
        conf['label_map'] = self.label_map
        conf['model_name'] = 'rnn-lstm-cnn'
        torch.save(conf, path)
        print("Model Saved as ", path)

    @staticmethod
    def load(path):
        conf = torch.load(path)

        model = RNN_LSTM_CNN(input_dim=conf['input_dim'], embedding_dim=conf['embedding_dim'],
                        bidirectional=conf['bidirectional'], hidden_dim=conf['lstm_hidden'],
                        output_dim=conf['output_dim'], dropout=conf['dropout'], pad_idx=conf['pad_idx'],
                        loss_func=conf['loss_func_name'], fc_hidden2=conf['fc_hidden2'], cnn_layer1=conf['cnn_layer_1'],
                        cnn_layer2=conf['cnn_layer_2'], fc_hidden1=conf['fc_hidden1'], k1_size=conf['k1_size'],
                        s1_size=conf['s1_size'], k2_size=conf['k2_size'], s2_size=conf['s2_size'],
                        cnn_dp=conf['cnn_dp'])
        model.label_map = conf['label_map']

        keys_to_drop = ['input_dim', 'fc_hidden1', 'fc_hidden2', 'input_dim', 'embedding_dim', 'bidirectional',
                        'num_layers', 'output_dim',
                        'dropout', 'pad_idx', 'loss_func_name', 'label_map', 'model_name', 'cnn_layer_1', 'cnn_layer_2',
                        'k1_size', 's1_size', 'k2_size', 's2_size', 'cnn_dp', 'lstm_hidden', 'cnn_dropout']

        for key in keys_to_drop:
            if key in conf.keys():
                conf.pop(key)

        model.load_state_dict(conf)

        return model
