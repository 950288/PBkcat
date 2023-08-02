import json
import model
import torch

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

    model = model.KcatPrediction(args, device).to(device)
    model.load_state_dict(torch.load(path))

    print(model.summary())


    
    

