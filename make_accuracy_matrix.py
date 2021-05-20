import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("data_points_pr_acc_r.txt", "rb") as fp:
#with open("lin-tr-exp/causal-copy/data_points.txt", "rb") as fp:
    data_points = pickle.load(fp)

#print(data_points)
n_models = 10
max_trained_depth = 11
max_test_depth = 20

for model_type in ['rnn', 'lstm', 'transformer']:
    model_title = 'Transformer' if model_type=='transformer' else 'LSTM' if model_type=='lstm' else 'RNN'

    accuracies = np.zeros((max_test_depth, max_trained_depth))
    successes = np.zeros((max_test_depth, max_trained_depth))

    for res in data_points:
        if res['model_type'] == model_type:
            accuracies[res['test_depth']-1][res['max_trained_depth']-1] += res['accuracy']
            successes[res['test_depth']-1][res['max_trained_depth']-1] += 1 if res['accuracy']==1 else 0

    accuracies /= n_models
    successes /= n_models
    print(accuracies)
    print(successes)

    fig, ax = plt.subplots(1,1)

    #img = ax.imshow(accuracies, origin='lower', extent=[0.5,max_trained_depth+0.5,0.5,max_test_depth+0.5])
    img = ax.imshow(accuracies, origin='lower', extent=[0.5,max_trained_depth+0.5,0.5,max_test_depth+0.5])

    ax.set_xticks(list(range(1, max_trained_depth+1)))
    ax.set_yticks(list(range(1, max_test_depth+1)))

    fig.colorbar(img)

    ax.set_xlabel('Max Training Depth')
    ax.set_ylabel('Depth')

    # add lines indicating training accuracy threshold 91 instead of 95
    if model_type == 'rnn':
        plt.axvline(x=7.5, color='red', linestyle='--')
    elif model_type == 'transformer':
        plt.axvline(x=9.5, color='red', linestyle='--')

    plt.title(f"{model_title} Accuracy Plot")
    plt.savefig('figures/accuracy_matrices_SVA_new/' + model_type + '_accuracy.png')

    plt.clf()
    fig, ax = plt.subplots(1,1)
    img = ax.imshow(successes, origin='lower', extent=[0.5,max_trained_depth+0.5,0.5,max_test_depth+0.5])

    ax.set_xticks(list(range(1, max_trained_depth+1)))
    ax.set_yticks(list(range(1, max_test_depth+1)))

    fig.colorbar(img)

    ax.set_xlabel('Max Training Depth')
    ax.set_ylabel('Depth')

    plt.title(f"{model_title} Success Plot")
    plt.savefig('figures/success_matrices_SVA_new/' + model_type + '_success.png')
