import json
import model
import torch
import random

if __name__ == "__main__":

    model_name = 'Kcat'

    with open('./model/output/' + model_name + '-args.json', 'r') as f:
        args = json.loads(f.read())

    path = './model/output/' + model_name + '.pth'

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU !!!')
    else:
        device = torch.device('cpu')
        print('The code uses CPU !!!')

    Kcatpredictor = model.KcatPrediction(args, device).to(device)
    Kcatpredictor.load_state_dict(torch.load(path))

    dir_input = './data/'
    compound_fingerprints = model.load_pickle(dir_input + 'compound_fingerprints.pickle')
    adjacencies = model.load_pickle(dir_input + 'adjacencies.pickle')
    proteins_local = model.load_pickle(dir_input + 'local_representations.pickle')
    # proteins_global = model.load_pickle(dir_input + 'global_representations.pickle')
    fingerprint_dict = model.load_pickle(dir_input + 'fingerprint_dict.pickle')
    args['len_fingerprint'] = len(fingerprint_dict)
    Kcat = model.load_pickle(dir_input + 'Kcats.pickle')
    Kcat = torch.LongTensor(Kcat)

    if not (len(compound_fingerprints) == len(adjacencies) == len(proteins_local) == len(Kcat)):
        print('The length of compound_fingerprints, adjacencies and proteins are not equal !!!')
        exit()

    dataset = list(zip(compound_fingerprints, adjacencies, proteins_local, Kcat))
    random.seed(2333)
    random.shuffle(dataset)
    dataset_train, dataset_ = model.split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = model.split_dataset(dataset_, 0.5)

    for data in dataset_train[:100]:
        pre = Kcatpredictor(data[:3])
        print(pre[0][0], data[3])


    
    

