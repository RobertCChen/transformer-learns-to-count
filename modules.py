import torch
import math
from collections import OrderedDict
from fast_transformers.masking import LengthMask, TriangularCausalMask
from fast_transformers.builders import TransformerEncoderBuilder, RecurrentEncoderBuilder

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, i_start=0):
        pos_embedding =  self.pe[:, i_start: i_start+x.size(1), :]
        pos_embedding = torch.repeat_interleave(pos_embedding, x.shape[0], dim=0)
        x =  torch.cat([x, pos_embedding], dim=2)
        return self.dropout(x)

class SequencePredictorRecurrentTransformer(torch.nn.Module):
    def __init__(self, d_model, n_classes, 
                 sequence_length=5000,
                 attention_type="causal-linear", 
                 n_layers=1, 
                 n_heads=1,
                 d_query=4,
                 dropout=0.0, 
                 softmax_temp=None,
                 attention_dropout=0.0,):
        super(SequencePredictorRecurrentTransformer, self).__init__()
        
        self.pos_embedding = PositionalEncoding(
            d_model//2,
            max_len=sequence_length
        )
        self.value_embedding = torch.nn.Embedding(
            n_classes,
            (d_model+1)//2
        )
        self.builder_dict = OrderedDict({
            "attention_type": attention_type,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "feed_forward_dimensions": n_heads*d_query*4,
            "query_dimensions": d_query,
            "value_dimensions": d_query,
            "dropout": dropout,
            "softmax_temp": softmax_temp,
            "attention_dropout": attention_dropout,
        })
        self.transformer = RecurrentEncoderBuilder.from_dictionary(
            self.builder_dict,
            strict=True
        ).get() # takes in batch x d_model
        hidden_size = n_heads*d_query
        self.predictor = torch.nn.Linear(
            hidden_size,
            n_classes
        )
        self.hidden_state = None # Would be tensor corresponding to hdn state after forward call
        def record_hdn(m, input_, output):
            self.hidden_state.append(output[0])
        self.transformer.layers[-1].attention.register_forward_hook(record_hdn)
        
    def forward(self, x, i_start=0, prev_state=None, return_state=False): # x is sequence_length x batch_size
        # i_start is how far into the whole sequence you are in. Used for positional encoding.
        x = x.view(x.shape[0], -1)
        x = self.value_embedding(x).transpose(1, 0)
        # print('xshape  here', x.shape)
        x = self.pos_embedding(x, i_start=i_start) # x is now batch_size x sequence_length x hdn_size
        y_hat = []
        self.hidden_state = []
        state = prev_state
        for i in range(x.size(1)):
            out, state = self.transformer(x[:, i, :], state=state)
            y_hat.append(out) # batch x d_model
        y_hat = torch.stack(y_hat, 1)
        y_hat = self.predictor(y_hat)
        self.hidden_state = torch.stack(self.hidden_state, 1)
        if return_state:
            return y_hat, state
        else:
            return y_hat


"""
class SequencePredictorTransformer(torch.nn.Module):
    def __init__(self, d_model, sequence_length, n_classes,
                 attention_type="causal-linear", n_layers=1, n_heads=1,
                 d_query=3, dropout=0.1, softmax_temp=None,
                 attention_dropout=0.1,
                 bits=32, rounds=4,
                 chunk_size=32, masked=True):
        super(SequencePredictorTransformer, self).__init__()

        self.pos_embedding = PositionalEncoding(
            d_model//2,
            max_len=sequence_length
        )
        self.value_embedding = torch.nn.Embedding(
            n_classes,
            (d_model + 1) // 2
        )
        self.builder_dict = OrderedDict({
            "attention_type": attention_type,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "feed_forward_dimensions": n_heads*d_query*4,
            "query_dimensions": d_query,
            "value_dimensions": d_query,
            "dropout": dropout,
            "softmax_temp": softmax_temp,
            "attention_dropout": attention_dropout,
            "bits": bits,
            "rounds": rounds,
            "chunk_size": chunk_size,
            "masked": masked
        })
        self.transformer = TransformerEncoderBuilder.from_dictionary(
            self.builder_dict,
            strict=True
        ).get()
        hidden_size = n_heads*d_query
        self.predictor = torch.nn.Linear(
            hidden_size,
            n_classes
        )

    def forward(self, x): # x is sequence_length x batch_size
        x = x.view(x.shape[0], -1)
        x = self.value_embedding(x).transpose(1, 0)
        x = self.pos_embedding(x)
        triangular_mask = TriangularCausalMask(x.shape[1], device=x.device)
        y_hat = self.transformer(x, attn_mask=triangular_mask) # batch x seq_len x hdn_size
        y_hat = self.predictor(y_hat)
        return y_hat
"""

class SequencePredictorRNN(torch.nn.Module):
    def __init__(self, d_model, n_classes,
                 n_layers=1, 
                 dropout=0.0, 
                 rnn_type="lstm"):
        super(SequencePredictorRNN, self).__init__()
        self.value_embedding = torch.nn.Embedding(
            n_classes,
            d_model,
        )
        self.rnn_type = rnn_type
        hidden_size = d_model
        rnn_ = torch.nn.LSTM if rnn_type == 'lstm' else torch.nn.RNN
        self.rnn = rnn_(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.predictor = torch.nn.Linear(
            hidden_size,
            n_classes
        )
        self.hidden_state = None
        def record_hdn(m, input_, output):
            if self.rnn_type == 'lstm':
                self.hidden_state.append(output[1][0])
            else:
                assert self.rnn_type == 'rnn'
                self.hidden_state.append(output[1])
        self.rnn.register_forward_hook(record_hdn)

    def forward(self, x, prev_state=None, return_state=False): # x is sequence_length x batch_size
        x = x.view(x.shape[0], -1)
        x = self.value_embedding(x).transpose(1, 0)
        # x is batch x seqlen x hdnsize
        state = prev_state
        y_hat = []
        self.hidden_state = []
        for i in range(x.size(1)):
            out, state = self.rnn(x[:, i:i+1, :], state)
            y_hat.append(out)
        y_hat = torch.cat(y_hat, dim=1)
        self.hidden_state = torch.cat(self.hidden_state, dim=1)
        y_hat = self.predictor(y_hat)
        if return_state:
            return y_hat, state
        else:
            return y_hat
