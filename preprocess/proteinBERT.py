# pip install protein_bert

import proteinbert
from proteinbert import load_pretrained_model
import numpy as np
import pickle
import json

pretrained_model_generator, input_encoder = load_pretrained_model(local_model_dump_dir = "./preprocess" , local_model_dump_file_name = 'epoch_92400_sample_23500000.pkl')

model = pretrained_model_generator.create_model(seq_len = 16)

global_representations = list()
local_representations = list()

with open('./data/Kcat_combination_0918.json', 'r') as infile :
    Kcat_data = json.load(infile)

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(dictionary), file)

def save_array(array, filename):
    with open(filename, 'wb') as file:
        pickle.dump(array, file)

for i , data in enumerate(Kcat_data) :
    input_ids = input_encoder.encode_X(data['Sequence'] , 16)
    local_representation , global_representation = model.predict(input_ids)
    # global_representations.append(global_representation)
    local_representations.append(local_representation)
    if i % 100 == 0 :
        print(i + 1 , '/' , len(Kcat_data) , end = '\n')

save_array(global_representations, './data/global_representations.pickle')

print('global_representations saved successfully!')