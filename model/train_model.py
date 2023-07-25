import model
import pickle
import numpy as np
import torch

if __name__ == "__main__":

    model_name = 'Kcat'

    args = {
        'dim' : 10,
        'layer_output' : 3,
        'layer_gnn' : 3,
        'layer_dnn' : 3,
    }

    file_model = './model/output/' + model_name
    file_MAEs  = './model/output/' + model_name + '-MAEs.txt'
    file_args  = './model/output/' + model_name + '-args'

    dir_input = './data/'
    compound_fingerprints = model.load_pickle(dir_input + 'compound_fingerprints.pickle')
    adjacencies = model.load_pickle(dir_input + 'adjacencies.pickle')
    proteins = model.load_pickle(dir_input + 'global_representations.pickle')
    fingerprint_dict = model.load_pickle(dir_input + 'fingerprint_dict.pickle')
    args['len_fingerprint'] = len(fingerprint_dict)

    dataset = list(zip(compound_fingerprints, adjacencies, proteins))
    print('The lenth of dataset: %d' % len(dataset))
    # shuffle dataset list ---------------
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
    model = model.KcatPrediction(args).to(device)
    # trainer = model.Trainer(model)
    # tester = model.Tester(model)

    """Output files."""
    with open(file_MAEs, 'w') as f:
        f.write(MAEs + '\n')
    with open(file_args, 'w') as f:
        pickle.dump(args, f)



