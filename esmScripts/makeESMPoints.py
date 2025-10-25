from transformers import AutoTokenizer, AutoModel
import torch
import csv
import pandas as pd
import numpy as np

print(torch.__version__)
print(torch.cuda.is_available())

# Load model (esm2_t6_8M_UR50D = smallest 6-layer model)
model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example: single sequence
def get_mean_embedding(model, tokenizer, seq):
    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    layer_embeddings = []
    for i, h in enumerate(out.hidden_states):
        # Mean-pool across residues
        emb = h.mean(dim=1).squeeze().detach().cpu()
        layer_embeddings.append(emb)

    return torch.stack(layer_embeddings)



with open('training/all_points.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    
    embeddings = []
    for row in reader:
        if counter > 0:
            embeddings.append(get_mean_embedding(model, tokenizer, row[0]))
        counter += 1


x = torch.stack(embeddings)
print(x.shape)

torch.save(x, "esm6layer.pt")

model_name2 = "facebook/esm2_t12_35M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name2)
model2 = AutoModel.from_pretrained(model_name2)
with open('training/all_points.csv', newline='') as f:
    reader = csv.reader(f)
    counter = 0
    
    embeddings = []
    for row in reader:
        if counter > 0:
            embeddings.append(get_mean_embedding(model2, tokenizer, row[0]))
        counter += 1
x = torch.stack(embeddings)
print(x.shape)

torch.save(x, "esm12layer.pt")

