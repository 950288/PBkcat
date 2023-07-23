from proteinbert import load_pretrained_model
import pickle
import json
# from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

pretrained_model_generator, input_encoder = load_pretrained_model(local_model_dump_dir = ".\\preprocess\\" , local_model_dump_file_name = 'epoch_92400_sample_23500000.pkl')

model = pretrained_model_generator.create_model(seq_len = 512)
# model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(seq_len = 512))

global_representations = []

with open('./Kcat_combination_0918.json', 'r') as infile :
    Kcat_data = json.load(infile)

for data in Kcat_data :
    input_ids = input_encoder.encode_X(data['Sequence'] , 512)
    _ , global_representation = model.predict(input_ids)
    global_representations.append(global_representation)
    # local_representations, global_representation = model.predict(input_ids)
    print(len(global_representations) , '/' , len(Kcat_data) , end = '\n')

# print(local_representations.shape) # (20, 512, 26)
# print(global_representation.shape) # (20, 8943)

# with open('local_representations.pkl', 'wb') as f:
#     pickle.dump(local_representations, f)

with open('global_representations.pkl', 'wb') as f:
  pickle.dump(global_representations, f)
