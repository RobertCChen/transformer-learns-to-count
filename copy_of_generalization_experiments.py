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
    if args.model_type == "transformer":
        model = SequencePredictorRecurrentTransformer(
                    d_model=args.d_model, n_classes=5,
                    sequence_length=args.sequence_length,
                    attention_type=args.attention_type,
                    n_layers=args.n_layers,
                    n_heads=args.n_heads,
                    d_query=args.d_model, # used to be d_query
                    dropout=args.dropout,
                    softmax_temp=None,
                    attention_dropout=args.attention_dropout,
                )
    else:
        model = SequencePredictorRNN(
                    d_model=args.d_model, n_classes=5,
                    n_layers=args.n_layers,
                    dropout=args.dropout,
                    rnn_type=args.model_type
                )
    print(f"Created model:\n{model}")
    model.to(device)
    print("Number of epochs model was trained on: ",torch.load(args.continue_from, map_location=device)['epoch'])
    model.load_state_dict(torch.load(args.continue_from, map_location=device)['model_state'])

    def format_preds(x, y, preds, mask):
        n = len(x)
        n_dig = math.floor(math.log10(n)) + 1
        nums = []
        for p_dig in range(n_dig):
            nums.append( "# |" + "".join([str((i//10**p_dig)%10) for i in range(n)]) + "\n")
        nums = "".join(nums[::-1])
        xs = "x |" + "".join([str(int(v)) for v in x]) + "\n"
        ys = "y |" + "".join([elt if mask[i] == 1 else '?' for i, elt in enumerate([str(int(v)) for v in y])]) + "\n"
        yh = "yh|" + "".join([elt if mask[i] == 1 else '?' for i, elt in enumerate([str(int(v)) for v in preds])]) + "\n"
        return nums + xs + ys + yh

    acc_list = []
    max_acc = None
    for stack_size in range(1, 64):
        x, y, m = SubjectVerbAgreement.get_seq(stack_size)
        # print(x.shape, y.shape, m.shape)
        model.eval()
        yhat = model(x.unsqueeze(1))
        hdn = model.hidden_state # batch x seq x hdn
        loss, acc = loss_fn(y.unsqueeze(1), yhat, m.unsqueeze(1))
        acc_list.append((stack_size, acc))
        if acc == 1:
            max_acc = stack_size
    print("Highest perfect score at depth:", max_acc)
    plot_hidden_state_2d(np.array(acc_list), pca=False)

    stack_size = 7 # Change this value to test longer / shorter sequences
    x, y, m = SubjectVerbAgreement.get_seq(stack_size)
    model.eval()
    yhat = model(x.unsqueeze(1))
    hdn = model.hidden_state # batch x seq x hdn
    loss, acc = loss_fn(y.unsqueeze(1), yhat, m.unsqueeze(1))
    print("Model loss: ", loss)
    print("Model accuracy: ", acc)
    print(format_preds(x, y, torch.argmax(yhat, dim=2)[0], m))
    plot_hidden_state_2d(hdn[0].detach().cpu().numpy(), pca=True)

    """
    Run
        rnn:    python generalization_experiments.py --model_type=rnn --d_model=3 --continue_from=model_storage/model_rnn
        lstm:   python generalization_experiments.py --model_type=lstm --d_model=3 --continue_from=model_storage/model_lstm
        transformer:   python generalization_experiments.py --model_type=transformer --d_model=8 --continue_from=model_storage/model_transformer
    """
if __name__ == "__main__":
    main()
