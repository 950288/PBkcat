import model
import torch
import random
import timeit
import json

if __name__ == "__main__":

    """model_name used to save the model and MAEs"""

    model_name = 'Kcat_811'

    """hyperparameters"""
    args = {
        "dim" : 20,
        "layer_output" : 3,
        "layer_gnn" : 3,
        "layer_dnn" : 3,
        "lr" : 1e-4,
        "weight_decay": 1e-6,
        "epoch" : 100,
        "random_seed" : 810,
    }

    """dirs used to save the model and MAEs"""
    file_model = './model/output/' + model_name
    file_MAEs  = './model/output/' + model_name + '-MAEs.csv'
    file_args  = './model/output/' + model_name + '-args.json'

    dir_input = './data/'
    """load data"""
    compound_fingerprints = model.load_pickle(dir_input + 'compound_fingerprints.pickle')
    adjacencies = model.load_pickle(dir_input + 'adjacencies.pickle')
    proteins_local = model.load_pickle(dir_input + 'local_representations.pickle') # in shape of (n_proteins, 20, 20)
    proteins_global = model.load_pickle(dir_input + 'global_representations.pickle') # in shape of (n_proteins, 20, 20)
    fingerprint_dict = model.load_pickle(dir_input + 'fingerprint_dict.pickle')
    args['len_fingerprint'] = len(fingerprint_dict)
    Kcat = torch.FloatTensor(model.load_pickle(dir_input + 'Kcats.pickle'))

    """check the length of the data"""
    if not (len(compound_fingerprints) == len(adjacencies) == len(proteins_local) == len(proteins_global) == len(Kcat)):
        print('The length of compound_fingerprints, adjacencies and proteins are not equal !!!')
        exit()

    dataset = list(zip(compound_fingerprints, adjacencies, proteins_local, proteins_global, Kcat))
    random.seed(args["random_seed"])
    random.shuffle(dataset)
    """split the dataset into train, dev and test set"""
    dataset_train, dataset_ = model.split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = model.split_dataset(dataset_, 0.5)

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU !!!')
    else:
        device = torch.device('cpu')
        print('The code uses CPU !!!')

    Kcatpredictor = model.KcatPrediction(args, device).to(device)
    trainer = model.Trainer(Kcatpredictor)
    tester = model.Tester(Kcatpredictor)

    """ Output files"""
    with open(file_args, 'w') as f:
        f.write(str(json.dumps(args)) + '\n')

    """ Start training """
    print('Training...')
    MAEs = []
    start = timeit.default_timer()

    """ training and testing """
    for epoch in range(0, args["epoch"]):
        print('Epoch: %d / %d' % (epoch + 1, args["epoch"]))
        LOSS_train, RMSE_train, R2_train, Lr = trainer.train(dataset_train)
        LOSS_dev, RMSE_dev, R2_dev = tester.test(dataset_dev)
        end = timeit.default_timer()
        time = end - start
        MAE = [epoch+1, time, LOSS_train, RMSE_train, R2_train, 
                            LOSS_dev,  RMSE_dev,  R2_dev,  Lr]
        MAEs.append(MAE)

        """save model and MAEs every 20 epoch"""
        if (epoch) % 20 == 0:
            torch.save(Kcatpredictor.state_dict(), file_model +'_'+ str(epoch) + ".pth")
        with open(file_MAEs, 'w') as f:
            f.write('epoch,time,LOSS_train,RMSE_train,R2_train,LOSS_dev,RMSE_dev,R2_dev,Lr\n')
            for MAE in MAEs:
                f.write(str(MAE)[1:-1] + '\n')
        # print('MAEs saved to %s' % file_MAEs)

    """after training, test on the test set"""
    print("on the test set:")
    LOSS_test, RMSE_test, R2_test = tester.test(dataset_test)

    """Save the trained model."""
    torch.save(Kcatpredictor.state_dict(), file_model + ".pth")
    print('Model saved to %s' % file_model)

    """save MAEs as csv file"""
    with open(file_MAEs, 'w') as f:
        f.write('epoch,time,LOSS_train,RMSE_train,R2_train,LOSS_dev,RMSE_dev,R2_dev,Lr\n')
        for MAE in MAEs:
            f.write(str(MAE)[1:-1] + '\n')
    print('MAEs saved to %s' % file_MAEs)


