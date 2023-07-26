# pip install protein_bert

import proteinbert
from proteinbert import load_pretrained_model
import numpy as np
import pickle
import json

pretrained_model_generator, input_encoder = load_pretrained_model(local_model_dump_dir = "./preprocess" , local_model_dump_file_name = 'epoch_92400_sample_23500000.pkl')

model = pretrained_model_generator.create_model(seq_len = 512)

global_representations = list()

with open('./data/Kcat_combination_0918.json', 'rb+') as infile :
    Kcat_data = json.load(infile)

len_global_representations = 0
try :
    with open('./data/global_representations.pickle', 'rb') as infile :
        len_global_representations = len(pickle.load(infile))
finally :

    print('the length of global_representations is ' , len_global_representations)

    def save_array(array, filename):
        with open(filename, 'ab') as file:
            pickle.dump(array, file)
            print('global_representations saved successfully!')
        with open('./data/global_representations.pickle', 'rb+') as infile :
            print('the length of global_representations is ' , len(pickle.load(infile)))

    len_Kcat_data = len(Kcat_data)
    global_representations = list()
    for i , data in enumerate(Kcat_data[len_global_representations:]) :
        input_ids = input_encoder.encode_X(data['Sequence'] , 512)
        _ , global_representation = model.predict(input_ids)
        global_representations.append(global_representation)
        print(len_global_representations + i + 1 , '/' , len_Kcat_data , end = '\n')
        if len(global_representations) % 1 == 0 :
            save_array(global_representations , './data/global_representations.pickle')
            print('global_representations saved successfully!')
            global_representations = []

    save_array(global_representations , './data/global_representations.pickle')

    print('global_representations saved successfully!')