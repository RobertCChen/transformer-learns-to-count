import time
from collections import namedtuple
import sys

import numpy as np
import torch
import math
import torch.nn.functional as F

from radam import RAdam

import sklearn.decomposition


def add_optimizer_arguments(parser):
    parser.add_argument(
        "--optimizer",
        choices=["radam", "adam"],
        default="radam",
        help="Choose the optimizer"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Set the learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Set the weight decay"
    )

def get_optimizer(params, args):
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr)
    elif args.optimizer == "radam":
        return RAdam(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError("Optimizer {} not available".format(args.optimizer))

def add_transformer_arguments(parser):
    parser.add_argument(
        "--attention_type",
        type=str,
        choices=["full", "causal-linear", "reformer"],
        default="causal-linear",
        help="Attention model to be used"
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=1,
        help="Number of self-attention layers"
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=1,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--d_query",
        type=int,
        default=3,
        help="Dimension of the query, key, value embedding"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout to be used for transformer layers"
    )
    parser.add_argument(
        "--softmax_temp",
        type=float,
        default=None,
        help=("Softmax temperature to be used for training "
              "(default: 1/sqrt(d_query))")
    )
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0.0,
        help="Dropout to be used for attention layers"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=32,
        help="Number of planes to use for hashing for reformer"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=32,
        help="Number of queries in each block for reformer"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="Number of rounds of hashing for reformer"
    )
    parser.add_argument(
        "--unmasked_reformer",
        action="store_false",
        dest="masked",
        help="If set the query can attend to itsself for reformer"
    )

    return parser

def add_auxiliary_arguments(parser):
    parser.add_argument(
        "--d_model",
        type=int,
        default=2,
        help="Set the hidden size for RNN / LSTM"
    )
    parser.add_argument(
        "--plot_hidden",
        type=bool,
        default=True,
        help="Plot hidden state?"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="Set the maximum sequence length"
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=3,
        help="Set the number of classes"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1, # was 100
        help="How many epochs to train for"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="How many samples to use together"
    )
    parser.add_argument(
        "--reduce_lr_at",
        type=int,
        default=30,
        help="At this epoch divide the lr by 10"
    )

    parser.add_argument(
        "--save_to",
        default=None,
        help="Set a file to save the models to."
    )
    parser.add_argument(
        "--continue_from",
        default=None,
        help="Load the model from a file"
    )
    parser.add_argument(
        "--save_frequency",
        default=1,
        type=int,
        help="Save every that many epochs"
    )
    parser.add_argument(
        "--model_type",
        default='lstm',
        choices=["transformer", "rnn", "lstm"],
        type=str,
        help="Select transformer, rnn, or lstm"
    )
    return parser

def print_transformer_arguments(args):
    print((
        "Transformer Config:\n"
        "    Attention type: {attention_type}\n"
        "    Number of layers: {n_layers}\n"
        "    Number of heads: {n_heads}\n"
        "    Key/Query/Value dimension: {d_query}\n"
        "    Transformer layer dropout: {dropout}\n"
        "    Softmax temperature: {softmax_temp}\n"
        "    Attention dropout: {attention_dropout}\n"
        "    Number of hashing planes: {bits}\n"
        "    Chunk Size: {chunk_size}\n"
        "    Rounds: {rounds}\n"
        "    Masked: {masked}"
    ).format(**vars(args)))

class EpochStats(object):
    def __init__(self, metric_names=[], freq=1, out=sys.stdout):
        self._start = time.time()
        self._samples = 0
        self._loss = 0
        self._metrics = [0]*len(metric_names)
        self._metric_names = metric_names
        self._out = out
        self._freq = freq
        self._max_line = 0

    def update(self, n_samples, loss, metrics=[]):
        self._samples += n_samples
        self._loss += loss*n_samples
        for i, m in enumerate(metrics):
            self._metrics[i] += m*n_samples

    def _get_progress_text(self):
        time_per_sample = (time.time()-self._start) / self._samples
        loss = self._loss / self._samples
        metrics = [
            m/self._samples
            for m in self._metrics
        ]
        text = "Loss: {} ".format(loss)
        text += " ".join(
            "{}: {}".format(mn, m)
            for mn, m in zip(self._metric_names, metrics)
        )
        if self._out.isatty():
            to_add = " [{} sec/sample]".format(time_per_sample)
            if len(text) + len(to_add) > self._max_line:
                self._max_line = len(text) + len(to_add)
            text += " " * (self._max_line-len(text)-len(to_add)) + to_add
        else:
            text += " time: {}".format(time_per_sample)
        return text

    def progress(self):
        if self._samples < self._freq:
            return
        text = self._get_progress_text()
        if self._out.isatty():
            print("\r" + text, end="", file=self._out)
        else:
            print(text, file=self._out, flush=True)
        self._loss = 0
        self._samples = 0
        self._last_progress = 0
        for i in range(len(self._metrics)):
            self._metrics[i] = 0
        self._start = time.time()

    def finalize(self):
        self._freq = 1
        self.progress()
        if self._out.isatty():
            print("", file=self._out)

def load_model(saved_file, model, optimizer, device):
    data = torch.load(saved_file, map_location=device)
    model.load_state_dict(data["model_state"])
    optimizer.load_state_dict(data["optimizer_state"])
    epoch = data["epoch"]
    return epoch

def save_model(save_file, model, optimizer, epoch):
    torch.save(
        dict(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch
        ),
        save_file.format(epoch)
    )

def loss_fn(y, y_hat, loss_mask):
    y_hat = y_hat.transpose(1, 0).contiguous()
    L, N, C = y_hat.shape
    l = torch.nn.functional.cross_entropy(
        y_hat.view(L*N, C),
        y.contiguous().view(L*N),
        reduction="none"
    ).view(L, N)
    # this means longer sequences have higher weight but it sounds ok
    l = (loss_mask * l).mean() / loss_mask.mean()
    accuracy = ((y == y_hat.argmax(dim=-1)).float() * loss_mask).mean() / loss_mask.mean()
    return l, accuracy.item()

def train(model, optimizer, dataloader, device):
    model.train()
    stats = EpochStats(["accuracy"])
    for i, (x, y, m) in zip(range(100), dataloader):
        x = x.to(device).t()
        y = y.to(device).t()
        m = m.to(device).t()
        optimizer.zero_grad()
        y_hat = model(x)
        l, acc = loss_fn(y, y_hat, m)
        l.backward()
        optimizer.step()
        stats.update(x.shape[1], l.item(), [acc])
        stats.progress()
    stats.finalize()

def evaluate(model, dataloader, device, return_accuracy=False):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_samples = 0

    with torch.no_grad():
        for i, (x, y, m) in zip(range(20), dataloader):
            x = x.to(device).t()
            y = y.to(device).t()
            m = m.to(device).t()
            y_hat = model(x)
            l, acc = loss_fn(y, y_hat, m)
            total_loss += x.shape[1] * l.item()
            total_acc += x.shape[1] * acc
            total_samples += x.shape[1]
    print(
        "Testing =>",
        "Loss:",
        total_loss/total_samples,
        "Accuracy:",
        total_acc/total_samples
    )
    if return_accuracy:
        return total_acc/total_samples
    else:
        return total_loss/total_samples

def extract_hidden_state(model, x, device):
    y_hat = model(x)
    return model.hidden_state, y_hat

import matplotlib.pyplot as plt
def plot_hidden_state_2d(points, pca=False, arrow_size=0.000, annotate=True):
    """
    points is seq_len x hidden_size
    """
    if pca:
        PCA = sklearn.decomposition.PCA(n_components=2)
        x = PCA.fit_transform(points)
    else:
        x = points[:,:2] # Truncate to first two dims
    plt.clf()
    for i in range(len(points) - 1):
        px = points[i][0]
        py = points[i][1]
        pdx = points[i+1][0] - px
        pdy = points[i+1][1] - py
        if i == 0:
            plt.plot(px, py, 'ro')
        plt.arrow(px, py, pdx, pdy, head_width=arrow_size, head_length=arrow_size, width=min(0.001, 0.01 * math.sqrt(pdx**2+pdy**2)))
        if annotate:
            plt.annotate(str(i+1), (px+pdx, py+pdy))
    plt.show()

