import proteinbert
from proteinbert import load_pretrained_model
import numpy as np
import pickle
import json

pretrained_model_generator, input_encoder = load_pretrained_model(local_model_dump_dir = ".\\preprocess\\" , local_model_dump_file_name = 'epoch_92400_sample_23500000.pkl')

model = pretrained_model_generator.create_model(seq_len = 512)

global_representations = list()

with open('./data/test.json', 'r') as infile :
    Kcat_data = json.load(infile)

for data in Kcat_data :
    input_ids = input_encoder.encode_X(data['Sequence'] , 512)
    _ , global_representation = model.predict(input_ids)
    (global_representation_a,global_representation_b) = global_representation.shape
    global_representation_c = 512-global_representation_a
    global_representation_1 = np.pad(global_representation,((0,global_representation_c),(0,0)),'constant',constant_values = (0,0))
    global_representations.append(global_representation_1)
    print(len(global_representations) , '/' , len(Kcat_data) , end = '\n')

np.save('./data/global_representations.npy', global_representations)