import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("data_points_primacy_recency_real.txt", "rb") as fp:
#with open("lin-tr-exp/causal-copy/data_points.txt", "rb") as fp:
    data_points = pickle.load(fp)

#print(data_points)
n_models = 10
max_verb_depth = 20
max_test_depth = 20

for model_type in ['lstm','rnn','transformer']: #add transf + rnn
    model_title = 'Transformer' if model_type=='transformer' else 'LSTM' if model_type=='lstm' else 'RNN'

    accuracies = np.zeros((max_test_depth, max_verb_depth))

    for res in data_points:
        if res['model_type'] == model_type:
            accuracies[res['test_depth']-1][res['verb_depth']] += res['accuracy']

    #accuracies /= n_models
    print(accuracies)

    # remove bottom row corresponding to ignored <eos>
    accuracies = accuracies[1:]

    fig, ax = plt.subplots(1,1)

    #img = ax.imshow(accuracies, origin='lower', extent=[0.5,max_verb_depth+0.5,0.5,max_test_depth+0.5])
    img = ax.imshow(accuracies, origin='lower', extent=[1.5,max_verb_depth+1.5,1.5,max_test_depth+0.5])

    ax.set_xticks(list(range(2, max_verb_depth+2)))
    ax.set_yticks(list(range(2, max_test_depth+1)))

    fig.colorbar(img)

    ax.set_xlabel('Verb position')
    ax.set_ylabel('Depth')

    # add line indicating max training depth
    plt.axvline(x=11.5, color='black', linestyle='--')

    plt.title(f"{model_title} Accuracy Per Verb")
    plt.savefig('figures/primacy_recency_matrices_SVA/' + model_type + '_accuracy.png')
