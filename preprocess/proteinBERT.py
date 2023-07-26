# pip install protein_bert

import proteinbert
from proteinbert import load_pretrained_model
import numpy as np
import pickle
import json

pretrained_model_generator, input_encoder = load_pretrained_model(local_model_dump_dir = "./preprocess" , local_model_dump_file_name = 'epoch_92400_sample_23500000.pkl')

model = pretrained_model_generator.create_model(seq_len = 128)

with open('./data/Kcat_combination_0918.json', 'rb+') as infile :
    Kcat_data = json.load(infile)

try :
    with open('./data/global_representations.pickle', 'wb') as infile :
        f.truncate(0)
except :
    pass

def save(data, filename):
    with open(filename, 'ab') as file:
        pickle.dump(data, file)

len_Kcat_data = len(Kcat_data)
for i , data in enumerate(Kcat_data) :
    input_ids = input_encoder.encode_X(data['Sequence'] , 512)
    _ , global_representation = model.predict(input_ids)
    # print(global_representation.shape)
    save(global_representation , './data/global_representations.pickle')
    print(i + 1 , '/' , len_Kcat_data , end = '\n')

print('global_representations saved successfully!')