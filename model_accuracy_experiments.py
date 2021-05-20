import argparse
import math
import sys
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from fast_transformers.masking import LengthMask, TriangularCausalMask
from fast_transformers.builders import TransformerEncoderBuilder
from utils import add_optimizer_arguments, get_optimizer, \
    add_transformer_arguments, print_transformer_arguments, \
    EpochStats, load_model, save_model, loss_fn, train, evaluate, add_auxiliary_arguments,\
    plot_hidden_state_2d, extract_hidden_state
from datasets import CopyTask, CountTask, CountTaskWithEOS
from modules import SequencePredictorRNN, SequencePredictorRecurrentTransformer
from constants import device
import pickle

def main(argv=None):
    print("Running on {}".format(device))
    parser = argparse.ArgumentParser(
        description="Train a transformer for a copy task"
    )
    add_optimizer_arguments(parser)
    add_transformer_arguments(parser)
    add_auxiliary_arguments(parser)
    args = parser.parse_args(argv)
    print("args:\n-----\n", args)

    data_points = []
    for model_type in ['rnn']:
        for max_trained_depth in range(1, 12):
            for test_depth in range(1, 21):
                for ii in range(10):
                    if model_type == "transformer":
                        model = SequencePredictorRecurrentTransformer(
                                    d_model=8, n_classes=3,
                                    sequence_length=args.sequence_length,
                                    attention_type=args.attention_type,
                                    n_layers=args.n_layers,
                                    n_heads=args.n_heads,
                                    d_query=8, # used to be d_query
                                    dropout=args.dropout,
                                    softmax_temp=None,
                                    attention_dropout=args.attention_dropout,
                                )
                    else:
                        model = SequencePredictorRNN(
                                    d_model=3 if model_type=='lstm' else 16, n_classes=3,
                                    n_layers=args.n_layers,
                                    dropout=args.dropout,
                                    rnn_type=model_type
                                )
                    print(f"Created model:\n{model}")
                    model.to(device)
                    model_name = "models_from_colab/RNN_models_anbn/model_storage/model_" + model_type + "_depth_" + str(max_trained_depth) + "_num_" + str(ii) + ".zip"
                    model.load_state_dict(torch.load(model_name, map_location=device)['model_state'])

                    stack_size = test_depth
                    x, y, m = CountTaskWithEOS.get_seq(stack_size, 1, 2, 0, 3 * stack_size)
                    model.eval()
                    yhat = model(x.unsqueeze(1))
                    hdn = model.hidden_state # batch x seq x hdn
                    loss, acc = loss_fn(y.unsqueeze(1), yhat, m.unsqueeze(1))
                    data_points.append({'model_type': model_type, 'max_trained_depth': max_trained_depth,
                                        'test_depth': test_depth, 'accuracy': acc})
    print("data points:")
    print(data_points)

    with open("data_points_rnn.txt", "wb") as fp:
        pickle.dump(data_points, fp)


    """
    Run
        python generalization_experiments.py --model_type=rnn --d_model=3 --continue_from=model_storage/model_rnn
    to test rnn generalization
    """
if __name__ == "__main__":
    main()
