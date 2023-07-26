if __name__ == "__main__":

    model_name = 'Kcat'

    with open('./model/output/' + model_name + '-args.json', 'r') as f:
        args = json.loads(f.read())