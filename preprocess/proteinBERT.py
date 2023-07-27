# pip install protein_bert

import proteinbert
from proteinbert import load_pretrained_model
import numpy as np
import pickle
import json

pretrained_model_generator, input_encoder = load_pretrained_model(local_model_dump_dir = "./preprocess" , local_model_dump_file_name = 'epoch_92400_sample_23500000.pkl')

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

sequences = []
max_len = 0
for i , data in enumerate(Kcat_data) :
    sequences.append(str(data['Sequence']))
    max_len = max(max_len , len(data['Sequence']))

max_len += 2

model = pretrained_model_generator.create_model(max_len)

step = 256
for i in range(0, len(sequences), step):
    print(i, '/' , len(sequences))
    sequences_ = sequences[i:i+step]
    input_ids = input_encoder.encode_X(sequences_, max_len)
    # local_representations_ , global_representations_ = model.predict(input_ids, batch_size=16)
    if len(local_representations):
        # local_representations = np.concatenate((local_representations, local_representations_), axis=0)
        global_representations = np.concatenate((global_representations, global_representations_), axis=0)
    else:
        # local_representations = local_representations_
        global_representations = global_representations_

# print(local_representations.shape, global_representations.shape)
print(global_representations.shape)
save_array(global_representations, './data/global_representations.pickle')
# save_array(local_representations, './data/local_representations.pickle')

print('saved successfully!')