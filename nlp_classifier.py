import torch
from transformers import BertTokenizer, AutoTokenizer, AutoModel
from transformers import RobertaForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from transformers import BertConfig
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, OneCycleLR, ExponentialLR
from dataset import CustomDataset
from models.text_models.bert import BertForSequenceClassification
from models.text_models.rnn_lstm import RNN_LSTM, save_model
from models.text_models.deberta_v3 import DebertaV3
from base_tokenizer import Basic_tokenizer

base_models = {
    'bert': [BertForSequenceClassification, 'bert-base-uncased'],
    'roberta': [RobertaForSequenceClassification, 'roberta-base'],
    'lstm': [RNN_LSTM, 'lstm'],
    'deberta': [DebertaV3, 'microsoft/deberta-v3-base'],
    'cnn': None
}


class NLPClassifier:
    def __init__(self, base_model='lstm', num_labels=1, problem_type='single_label_classification',
                 train_split_rate=0.8):
        self.tokenizer_base = None
        self.num_labels = None
        self.base_model = base_model
        self.problem_type = problem_type
        self.moving_average = None
        self.update_interval = 10  # Update model parameters every 10 samples
        self.max_length = 512
        self.early_stopping_patience = 10
        self.save_path = 'best_weights'
        self.train_split = train_split_rate
        self.config = None

    @staticmethod
    def load(model_path, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        obj = NLPClassifier()
        if os.path.isdir(model_path):
            obj.model = AutoModel.from_pretrained(model_path)
            obj.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            conf = torch.load(model_path)
            model = RNN_LSTM(conf['input_dim'], conf['embedding_dim'], conf['bidirectional'], conf['lstm_hidden'],
                             conf['num_layers'], conf['output_dim'], conf['dropout'], conf['pad_idx'],
                             conf['fc_hidden1'], conf['fc_hidden2'], conf['loss_func_name'])

            keys_to_drop = ['lstm_hidden', 'fc_hidden1', 'fc_hidden2', 'input_dim', 'embedding_dim', 'bidirectional',
                            'num_layers', 'output_dim',
                            'dropout', 'pad_idx', 'loss_func_name']
            for key in keys_to_drop:
                conf.pop(key)

            model.load_state_dict(conf)

            obj.model = model
            tokenizer_path = model_path.replace(".pth", "_tokenizer")
            obj.tokenizer = Basic_tokenizer.load(tokenizer_path)

        return obj

    def load_model(self, base_model, problem_type, num_labels, train_dataset=None, min_freq=1):
        if base_model not in base_models.keys():
            print("Model not supported")
            return None

        if 'lstm' in base_model and train_dataset:
            custom_tokenizer = Basic_tokenizer(train_dataset, min_freq=min_freq)
            vocab_size = len(custom_tokenizer.vocab.get_stoi().keys())
            self.model = base_models[base_model][0](vocab_size, embedding_dim=300, bidirectional=True, hidden_dim=128,
                                                    num_layers=2, output_dim=num_labels, dropout=0.3,
                                                    pad_idx=custom_tokenizer.vocab['<pad>'],
                                                    fc_hidden1=128, fc_hidden2=64)

            self.tokenizer = custom_tokenizer.tokenize
            self.tokenizer_base = custom_tokenizer

            print("LSTM type model is loaded")
            return True

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_models[base_model][1])
            self.model = base_models[base_model][0].from_pretrained(base_models[base_model][1], num_labels=num_labels,
                                                                    problem_type=problem_type)
            if problem_type == 'regression':
                self.model.config.hidden_dropout_prob = 0
                self.model.config.attention_probs_dropout_prob = 0
                self.model.config.classifier_dropout = 0

            print("{} type model loaded".format(base_models[base_model][1]))
            return True

        except Exception as e:
            print("Error in loading model ", e)
            return None

    def train(self, input_texts, labels, epochs=10, batch_size=32, lr=2.5e-5, scheduler=None):
        print("Preprocessing")
        dataset_provided = False
        # input_ids = [self.tokenizer.encode(text, return_tensors='pt')[0] for text in input_texts]

        if isinstance(labels[0], str):
            label_transform = preprocessing.LabelEncoder()
            label_transform.fit(labels)
            labels = label_transform.transform(labels)

            np.save('label_classes.npy', label_transform.classes_)

            self.num_classes = len(label_transform.classes_)
            print("Labels size: ", self.num_classes)

        if isinstance(input_texts, CustomDataset) and isinstance(labels, CustomDataset):
            train_dataset = input_texts
            valid_dataset = labels

            train_loader = DataLoader(train_dataset, batch_size=batch_size)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
            self.num_classes = len(np.unique(valid_dataset.label))
            dataset_provided = True

        if not dataset_provided:
            train_size = int(len(input_texts) * self.train_split)
            train_ids = input_texts[:train_size]
            train_labels = labels[:train_size]
            train_labels = torch.tensor(train_labels)
            train_dataset = CustomDataset(train_ids, train_labels)

            valid_ids = input_texts[train_size:]
            valid_labels = torch.tensor(labels[train_size:])

            if self.problem_type == 'regression':
                train_labels = train_labels.to(torch.float32)
                valid_labels = valid_labels.to(torch.float32)

            valid_dataset = CustomDataset(valid_ids, valid_labels)

            train_loader = DataLoader(train_dataset, batch_size=batch_size)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

        loaded = self.load_model(self.base_model, self.problem_type, self.num_classes, train_dataset)

        if loaded is None:
            return None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        valid_num = len(valid_loader) * batch_size
        print("Training start")
        print("Training samples: ", len(train_loader) * batch_size)
        print("Validation samples: ", valid_num)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        # add scheduler
        if scheduler:
            total_steps = len(train_loader) * epochs
            scheduler = self.get_scheduler(optimizer, scheduler, lr=lr, total_steps=total_steps)
        best_valid_loss = 99999
        early_stopping_counter = 0
        self.model.train()
        for epoch in range(epochs):
            print("Epoch: ", epoch)
            valid_loss = 0
            valid_acc = 0
            self.model.train()
            for input_ids, labels in tqdm(train_loader):
                input_ids = self.tokenizer(input_ids, return_tensors="pt", padding=True, max_length=self.max_length,
                                           truncation=True)
                optimizer.zero_grad()

                if isinstance(input_ids, dict):
                    input_ids['input_ids'] = input_ids['input_ids'].to(device)
                else:
                    input_ids = input_ids.to(device)

                labels = labels.to(device)

                outputs = self.model(**input_ids, labels=labels)
                train_loss = outputs.loss
                train_loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

            # Evaluation of the model
            self.model.eval()
            with torch.no_grad():
                for input_ids, labels in valid_loader:
                    input_ids = self.tokenizer(input_ids, return_tensors="pt", padding=True, max_length=self.max_length,
                                               truncation=True)

                    if isinstance(input_ids, dict):
                        input_ids['input_ids'] = input_ids['input_ids'].to(device)
                    else:
                        input_ids = input_ids.to(device)

                    labels = labels.to(device)

                    outputs = self.model(**input_ids, labels=labels)

                    valid_loss += outputs.loss.item()
                    if 'regression' not in self.problem_type:
                        valid_acc += (outputs.logits.argmax(-1) == labels).sum().item()

                print("Valid len: ", len(valid_loader), valid_acc)
                valid_loss /= len(valid_loader)
                valid_acc = valid_acc / valid_num
                print("Epoch: {}, Valid Loss: {}, Valid Acc: {}".format(epoch, valid_loss, valid_acc))
                torch.cuda.empty_cache()

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    try:
                        self.model.save_pretrained(self.save_path)
                        self.tokenizer.save_pretrained(self.save_path)
                    except:
                        self.tokenizer_base.save(self.save_path + "_tokenizer")
                        if '.pth' not in self.save_path:
                            self.save_path += '.pth'
                        save_model(self.model, self.save_path)
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= self.early_stopping_patience:
                    print("Early stopping")
                    break

    def online_train(self, input_text, label):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        self.optimizer.zero_grad()
        outputs = self.model(input_ids, labels=torch.tensor([label]))
        loss = self.criterion(outputs.logits.view(-1, outputs.logits.size(-1)), torch.tensor([label]).view(-1))
        loss.backward()
        self.optimizer.step()
        # method
        self.scheduler.step()

        # 2
        if self.moving_average is None:
            self.moving_average = {param_name: param.detach().clone() for param_name, param in
                                   self.model.named_parameters()}
        else:
            for param_name, param in self.model.named_parameters():
                self.moving_average[param_name] = self.moving_average[param_name] * 0.9 + param.detach() * 0.1

        self.update_interval -= 1
        if self.update_interval == 0:
            for param_name, param in self.model.named_parameters():
                param.data.copy_(self.moving_average[param_name])
            self.update_interval = 10

    def get_scheduler(self, optimizer, s_type='onecycle', lr=3e-5, total_steps=None):
        if s_type == 'onecycle' and total_steps:
            scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=lr,
                                   anneal_strategy='linear', cycle_momentum=True)
        elif s_type == 'expo':
            scheduler = ExponentialLR(optimizer, gamma=0.999)
        elif s_type == 'cosine' and total_steps:
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        elif s_type == 'step':
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        return scheduler

    def predict(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        outputs = self.model(input_ids)
        return torch.argmax(outputs.logits).item()

    import random
    import torch
    from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

    def tune_bert(self, train_dataloader, valid_dataloader, num_epochs=4, batch_size=32, learning_rate=2e-5,
                  warmup_steps=0.1,
                  weight_decay=0.01, early_stopping_patience=3, save_model_path='best_model.bin'):
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1)
        criterion = torch.nn.CrossEntropyLoss()

        best_valid_loss = float('inf')
        early_stopping_counter = 0
        for epoch in range(num_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                logits = model(input_ids, attention_mask)
                loss = criterion(logits.view(-1, 2), labels.view(-1))
                loss.backward()
                optimizer.step()
                scheduler.step()

            model.eval()
            valid_loss = 0
            valid_acc = 0
            for step, batch in enumerate(valid_dataloader):
                input_ids, attention_mask, labels = batch
                logits = model(input_ids, attention_mask)
                loss = criterion(logits.view(-1, 2), labels.view(-1))
                valid_loss += loss.item()
                valid_acc += (logits.argmax(-1) == labels).sum().item()
            valid_loss /= len(valid_dataloader)
            valid_acc /= len(valid_dataloader)
            print("Epoch: {}, Valid Loss: {}, Valid Acc: {}".format(epoch, valid_loss, valid_acc))

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), save_model_path)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping")
                break

        # Random search of parameters
        for i in range(5):
            lr = 10 ** random.uniform(-5, -3)
            weight_decay = 10 ** random.uniform(-6, -3)
            print("Trying lr : ", lr, " weight_decay : ", weight_decay)
            model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1)
            criterion = torch.nn.CrossEntropyLoss()
            for epoch in range(num_epochs):
                model.train()
                for step, batch in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    input_ids, attention_mask, labels = batch
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits.view(-1, 2), labels.view(-1))
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                model.eval()
                valid_loss = 0
                valid_acc = 0
                for step, batch in enumerate(valid_dataloader):
                    input_ids, attention_mask, labels = batch
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits.view(-1, 2), labels.view(-1))
                    valid_loss += loss.item()
                    valid_acc += (logits.argmax(-1) == labels).sum().item()
                valid_loss /= len(valid_dataloader)
                valid_acc /= len(valid_dataloader)
                print("Epoch: {}, Valid Loss: {}, Valid Acc: {}".format(epoch, valid_loss, valid_acc))
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), save_model_path)
                    early_stopping_counter = 0
                    print("New best model found with lr: {}, weight_decay: {}".format(lr, weight_decay))
                    break
