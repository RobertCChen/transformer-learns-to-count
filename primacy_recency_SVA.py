import argparse
import math
import sys
from collections import OrderedDict
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader, IterableDataset
from fast_transformers.masking import LengthMask, TriangularCausalMask
from fast_transformers.builders import TransformerEncoderBuilder
from utils import add_optimizer_arguments, get_optimizer, \
    add_transformer_arguments, print_transformer_arguments, \
    EpochStats, load_model, save_model, loss_fn, train, evaluate, add_auxiliary_arguments,\
    plot_hidden_state_2d, extract_hidden_state
from datasets import CopyTask, CountTask, CountTaskWithEOS, SubjectVerbAgreement
from modules import SequencePredictorRNN, SequencePredictorRecurrentTransformer
from constants import device

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
    data_points_acc = []
    n_of_each_model = 10
    n_trials = 64
    for model_type in [ 'transformer','lstm','rnn',]: #add back transformers and rnn
        for test_depth in range(1, 21): # was 1, 32
            score_total = torch.zeros((test_depth,))
            for ii in range(n_of_each_model):
                if model_type == "transformer":
                    d_model = 16
                    model = SequencePredictorRecurrentTransformer(
                                d_model=d_model, n_classes=5,
                                sequence_length=args.sequence_length,
                                attention_type=args.attention_type,
                                n_layers=args.n_layers,
                                n_heads=args.n_heads,
                                d_query=d_model, # used to be d_query
                                dropout=args.dropout,
                                softmax_temp=None,
                                attention_dropout=args.attention_dropout,
                            )
                else:
                    d_model = 8
                    model = SequencePredictorRNN(
                                d_model=d_model, n_classes=5,
                                n_layers=args.n_layers,
                                dropout=args.dropout,
                                rnn_type=model_type
                            )
                print(f"Created model:\n{model}")
                model.to(device)
                model.load_state_dict(torch.load(f"models_from_colab/agreement_models/model_{model_type}_depth_11_num_{ii}.zip", map_location=device)['model_state'])
                stack_size = test_depth # Change this value to test longer / shorter sequences
                for i_trial in range(n_trials):
                    x, y, m = SubjectVerbAgreement.get_seq(stack_size)
                    model.eval()
                    yhat = model(x.unsqueeze(1)) # batch x seq x n_classes
                    model_preds = torch.argmax(yhat, dim=2)[0] # seq
                    score = torch.eq(model_preds, y)[stack_size:2*stack_size] # size stack_size. element 0 corresponds to center verb(recency), elt -1 last verb (primacy).
                    score_total += score
            score_avg = score_total / (n_trials * n_of_each_model)
            for verb_depth in range(test_depth-1): # -e to ignore <eos>
                data_points.append({'model_type':model_type,'test_depth':test_depth,
                    'verb_depth': verb_depth,
                    'accuracy': float(score_avg[verb_depth])
                })

    print("data points")
    print(data_points)
    with open("data_points_primacy_recency_real.txt", "wb") as fp:
        pickle.dump(data_points, fp)
    """
    Run
        rnn:    python generalization_experiments.py --model_type=rnn --d_model=3 --continue_from=model_storage/model_rnn
        lstm:   python generalization_experiments.py --model_type=lstm --d_model=3 --continue_from=model_storage/model_lstm
        transformer:   python generalization_experiments.py --model_type=transformer --d_model=8 --continue_from=model_storage/model_transformer
    """
if __name__ == "__main__":
    main()
