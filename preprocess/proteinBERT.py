# pip install protein_bert

import proteinbert
from proteinbert import load_pretrained_model
import numpy as np
import pickle
import json

pretrained_model_generator, input_encoder = load_pretrained_model(local_model_dump_dir = "./preprocess" , local_model_dump_file_name = 'epoch_92400_sample_23500000.pkl')

model = pretrained_model_generator.create_model(seq_len = 512)

global_representations = list()

with open('./data/Kcat_combination_0918.json', 'r') as infile :
    Kcat_data = json.load(infile)

with open('./data/global_representations.pickle', 'rb') as infile :
    global_representations = pickle.load(infile)

len_global_representations = len(global_representations)


def save_array(array, filename):
    with open(filename, 'a') as file:
        pickle.dump(array, file)


len_Kcat_data = len(Kcat_data)
global_representations = []
for data in Kcat_data[len_global_representations:] :
    input_ids = input_encoder.encode_X(data['Sequence'] , 512)
    _ , global_representation = model.predict(input_ids)
    global_representations.append(global_representation)
    print(len(global_representations) , '/' , len_Kcat_data , end = '\n')
    if len(global_representations) % 100 == 0 :
        save_array(global_representations , './data/global_representations.pickle')
        global_representations = []

save_array(global_representations , './data/global_representations.pickle')

print('global_representations saved successfully!')