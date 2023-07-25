import model
import pickle
import torch
import random
import numpy as np
import timeit
import json

if __name__ == "__main__":

    model_name = 'Kcat'

    args = {
        "dim" : 10,
        "layer_output" : 3,
        "layer_gnn" : 3,
        "layer_dnn" : 3,
        "lr" : 1e-3,
        "weight_decay": 1e-6,
        "iteration" : 100
    }

    file_model = './model/output/' + model_name
    file_MAEs  = './model/output/' + model_name + '-MAEs.txt'
    file_args  = './model/output/' + model_name + '-args.json'

    dir_input = './data/'
    compound_fingerprints = model.load_pickle(dir_input + 'compound_fingerprints.pickle')
    adjacencies = model.load_pickle(dir_input + 'adjacencies.pickle')
    proteins = model.load_pickle(dir_input + 'global_representations.pickle')
    fingerprint_dict = model.load_pickle(dir_input + 'fingerprint_dict.pickle')
    Kcat = model.load_pickle(dir_input + 'Kcats.pickle')
    Kcat = torch.from_numpy(np.array(Kcat))
    args['len_fingerprint'] = len(fingerprint_dict)

    dataset = list(zip(compound_fingerprints, adjacencies, proteins, Kcat))
    print('The lenth of dataset: %d' % len(dataset))
    random.shuffle(dataset)
    dataset_train, dataset_ = model.split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = model.split_dataset(dataset_, 0.5)

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU !!!')
    else:
        device = torch.device('cpu')
        print('The code uses CPU !!!')

    # torch.manual_seed(1234)
    Kcatpredictor = model.KcatPrediction(args).to(device)
    trainer = model.Trainer(Kcatpredictor)
    tester = model.Tester(Kcatpredictor)

    """Output files."""
    with open(file_args, 'w') as f:
        f.write(str(json.dumps(args)) + '\n')

    """Start training."""
    print('Training...')
    MAEs = []
    start = timeit.default_timer()
    for epoch in range(0, args["iteration"]):
        print('Epoch: %d / %d' % (epoch + 1, args["iteration"]))
        LOSS_train, RMSE_train, R2_train = trainer.train(dataset_train)
        LOSS_test, RMSE_test, R2_test = tester.test(dataset_dev)

        end = timeit.default_timer()
        time = end - start
        MAE = [epoch, time, LOSS_train, RMSE_train, R2_train, 
                            LOSS_test, RMSE_test, R2_test, R2_test]
        MAEs.append(MAE)

    """Save the trained model."""
    torch.save(Kcatpredictor.state_dict(), file_model)
    print('Model saved to %s' % file_model)
    """save MAEs"""
    with open(file_MAEs, 'w') as f:
        f.write(str(MAEs) + '\n')

        



