import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from scipy.special import expit

class attention(nn.Module):
    def __init__(self, hidden_size):
        super(attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(2*hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        batch_size, seq_len, feat_n = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seq_len, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2*self.hidden_size)

        x = self.linear1(matching_inputs)
        x = self.linear2(x)
        attention_weights = self.to_weight(x)
        attention_weights = attention_weights.view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context


class encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_ratio=0.3):
        super(encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.dropout = nn.Dropout(dropout_ratio)
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, feats):
        batch_size, seq_len, feat_n = feats.size()    
        feats = feats.view(-1, feat_n)
        feats = self.linear(feats)
        feats = self.dropout(feats)
        feats = feats.view(batch_size, seq_len, self.hidden_size) 

        output, hidden_state = self.rnn(feats)
        return output, hidden_state


class decoder(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, helper=None, dropout_ratio=0.3):
        super(decoder, self).__init__()

        self.hidden_size = hidden_size 
        self.output_size = output_size 
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.helper = helper

        self.embedding = nn.Embedding(vocab_size, word_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        self.rnn = nn.GRU(hidden_size+word_dim, hidden_size, batch_first=True)
        self.attention = attention(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)


    def forward(self, encoder_hidden, encoder_output, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = encoder_hidden.size()
        decoder_hidden = self.initialize_hidden(encoder_hidden)
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()  
        decoder_current_input_word = decoder_current_input_word.cuda() if torch.cuda.is_available() else decoder_current_input_word
        seq_logProb = []
        seq_predictions = []

        targets = self.embedding(targets) #  embeddings 
        _, seq_len, _ = targets.size()

        for i in range(seq_len-1): 
            threshold = self._get_teacher_learning_ratio(training_steps=tr_steps)
            current_input_word = targets[:, i] if random.uniform(0.05, 0.995) > threshold \
                else self.embedding(decoder_current_input_word).squeeze(1)

            # attention 
            context = self.attention(decoder_hidden, encoder_output)
            decoder_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            gru_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
            logprob = self.to_final_output(gru_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions


    def infer(self, encoder_hidden, encoder_output):
        _, batch_size, _ = encoder_hidden.size()
        decoder_hidden = self.initialize_hidden(encoder_hidden)
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()  
        decoder_current_input_word = decoder_current_input_word.cuda() if torch.cuda.is_available() else decoder_current_input_word
        seq_logProb = []
        seq_predictions = []
        assumption_seq_len = 28
        
        for i in range(assumption_seq_len-1):
            current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
            context = self.attention(decoder_hidden, encoder_output)
            decoder_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            gru_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
            logprob = self.to_final_output(gru_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions


    def initialize_hidden(self, encoder_hidden):
        if encoder_hidden is None:
            return None
        else:
            return encoder_hidden


    def _get_teacher_learning_ratio(self, training_steps):
        return (expit(training_steps/20 +0.85))

class MODELS(nn.Module):
    def __init__(self, encoder, decoder):
        super(MODELS, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, avi_feats, mode, target_sentences=None, tr_steps=None):
        encoder_outputs, (encoder_hidden, cell_state) = self.encoder(avi_feats)
        if mode == 'train':
            seq_logProb, seq_predictions = self.decoder( encoder_hidden,  encoder_outputs,
                 target_sentences, mode=mode , tr_steps = tr_steps)
        elif mode == 'inference':
            seq_logProb, seq_predictions = self.decoder.infer(encoder_hidden=encoder_hidden, encoder_output=encoder_outputs)
        else:
            raise KeyError('mode is not valid')
        return seq_logProb, seq_predictions

    




