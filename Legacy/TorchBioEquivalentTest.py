# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 18:36:36 2024 GMT-5

@author: Kwangho Baek baek0040@umn.edu dptm22203@gmail.com
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Configuration
batch_size = 1000
num_alternatives = 3
num_attributes = 4
num_epochs=300
lrate=0.02

# Sample data for all J alternatives: (batch_size, num_alternatives, num_attributes)
Z = torch.randn(batch_size, num_alternatives, num_attributes) #std random
X = 10 + 3 * Z
beta = torch.randn(num_attributes)
eps = 10 * torch.randn(batch_size, num_alternatives)  # Noise with mean 0 and std 3
utils = torch.matmul(X, beta) + eps
y = torch.argmax(utils, dim=1)  # Chosen alternative for each agent (shape: batch_size)
print(f"Synthesized Betas: {beta}")
print('Test starts')
print('')

#%% Model 1: All J alternatives (absolute attributes)
class MultinomialLogitModel(nn.Module):
    def __init__(self, num_attributes, num_alternatives):
        super(MultinomialLogitModel, self).__init__()
        self.linear = nn.Linear(num_attributes, 1, bias=False)  # One linear model per alternative

    def forward(self, x):
        # Apply the same linear transformation to each alternative
        utilities = self.linear(x)  # Shape: (batch_size, num_alternatives, 1)
        utilities = utilities.squeeze(-1)  # Remove the last dimension: (batch_size, num_alternatives)
        return utilities

#%% Model 2: J-1 alternatives (differences with respect to alternative 0)
# Step 1: Reorder X so that the chosen alternative is moved to the last row for each individual
def reorder_X_based_on_y(X, y):
    X_reordered = X.clone()  # Clone the original X to avoid modifying it
    y_reordered = y.clone()
    for i in range(batch_size):
        chosen_idx = y[i].item()  # Get the chosen alternative for individual i
        if chosen_idx==0: #if chosen is the first alt, push it to the last
            X_reordered[i]=torch.cat((X[i,1:,:],X[i,0,:].unsqueeze(0)),dim=0)
            y_reordered[i]=num_alternatives-1
    return X_reordered, y_reordered

# Reorder X so that the chosen alternative is always the last row for each individual
X_reordered, y_reordered = reorder_X_based_on_y(X, y)

# Step 2: Compute the differences between non-chosen alternatives and the chosen one
# After reordering, the last row of each individual's X is the chosen alternative
X_diff = X_reordered[:, 1:, :] - X_reordered[:, 0, :].unsqueeze(1)  # Compute differences with the first (always nonchosen) alt.

# Adjust labels to account for the deleted first alt
y_diff = y_reordered-1 

### Model: J-1 alternatives (differences with respect to the reordered chosen alternative)
class ReducedMultinomialLogitModel(nn.Module):
    def __init__(self, num_attributes):
        super(ReducedMultinomialLogitModel, self).__init__()
        self.linear = nn.Linear(num_attributes, 1,bias=False)  # One linear model for difference

    def forward(self, x_diff):
        # Compute the differences in utilities for J-1 alternatives
        utilities_diff = self.linear(x_diff)  # Shape: (batch_size, J-1, 1)
        utilities_diff = utilities_diff.squeeze(-1)  # Shape: (batch_size, J-1)
        return utilities_diff


#%% Training
# Instantiate the models
model_all = MultinomialLogitModel(num_attributes, num_alternatives)
model_diff = ReducedMultinomialLogitModel(num_attributes)

# Define the loss function (cross entropy) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_all = optim.Adam(model_all.parameters(), lr=lrate)
optimizer_diff = optim.Adam(model_diff.parameters(), lr=lrate)

# Prepare to store losses for plotting
losses_all = []
losses_diff = []

for epoch in range(num_epochs):
    # Clear gradients
    optimizer_all.zero_grad()
    optimizer_diff.zero_grad()

    ### Model 1: Multinomial Logit Model (all alternatives) ###
    logits_all = model_all(X)
    loss_all = criterion(logits_all, y)
    loss_all.backward()
    optimizer_all.step()
    losses_all.append(loss_all.item())  # Store loss

    ### Model 2: Reduced Multinomial Logit Model (J-1 alternatives) ###
    logits_diff = model_diff(X_diff)
    loss_diff = criterion(logits_diff, y_diff)
    loss_diff.backward()
    optimizer_diff.step()
    losses_diff.append(loss_diff.item())  # Store loss

# Print the parameters after optimization for Model 1
print("Parameters for Model 1 (all alternatives):")
for param in model_all.parameters():
    print(param.data)

# Print the parameters after optimization for Model 2
print("\nParameters for Model 2 (J-1 alternatives):")
for param in model_diff.parameters():
    print(param.data)

#%% Plot
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), losses_all, label='All Alternatives', color='b')
plt.plot(range(num_epochs), losses_diff, label='J-1 Alternatives', color='r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss for Pytorch-made Multinomial Logit Models')
plt.legend()
plt.grid(True)
plt.show()


#%% Compare Model1 with Biogeme

import pandas as pd
from biogeme.expressions import Beta, Variable
from biogeme import biogeme
from biogeme.models import loglogit
from biogeme.database import Database

# Generate the long-format data for Biogeme
def create_biogeme_data(X, y):
    data_list = []
    for i in range(batch_size):
        for j in range(num_alternatives):
            row = {
                'ID': i,  # Individual ID
                'alternative': j,  # Alternative number
                'chosen': 1 if y[i].item() == j else 0  # 1 if chosen alternative, 0 otherwise
            }
            # Add the attributes for the alternative
            for k in range(num_attributes):
                row[f'attr{k}'] = X[i, j, k].item()  # Attribute values for this alternative
            data_list.append(row)
    # Convert to a DataFrame
    df_biogeme = pd.DataFrame(data_list)
    return df_biogeme

#reindexing for them to start from 1
dfLong = create_biogeme_data(X, y)
dfLong.ID+=1
dfLong.alternative+=1

attrcols=['attr'+str(i) for i in range(num_attributes)]
def ppForBiogeme(dfLong,attrcols=['attr'+str(i) for i in range(num_attributes)]):
    dfWide= dfLong.pivot(index='ID', columns='alternative',values=attrcols)
    dfWide.columns = [f'{col[0]}_{int(col[1])}' for col in dfWide.columns]
    dfWide=dfWide.reset_index()
    choice=dfLong.loc[dfLong.chosen==1,['ID','alternative']]
    choice.columns=['ID','chosen']
    dfWide = pd.merge(dfWide, choice, on='ID')
    return dfWide

dfPP=ppForBiogeme(dfLong)

database = Database('mymodel',dfPP)

# Define the choice variable
chosen = Variable('chosen')

# Define betas for the attributes
betas = [Beta(f'BETA_{k}', 0.1, None, None, 0) for k in range(num_attributes)]

# Use a for loop to build the utility function
V={}
for i in range(1,num_alternatives + 1): #for each alts
    V[i]=0
    for k in range(num_attributes): #sum each attrs' utility contributions
        attrs = [Variable(f'attr{k}_{i}') for k in range(num_attributes)]
        V[i] += attrs[k] * betas[k]


# Define the logit model (with no availability restrictions)
logprob = loglogit(V, None, chosen)
# Create the Biogeme object
biogeme_model = biogeme.BIOGEME(database, logprob)
biogeme_model.modelName = ''
biogeme_model.generate_html = False  # Disable HTML report
biogeme_model.generate_pickle = False  # Disable pickle output
biogeme_model.save_iterations = False  # Disable saving of iteration logs (prevents .iter files)
biogeme_model.generate_toml = False  # Prevent creation of the .toml configuration file

# Estimate the model
results = biogeme_model.estimate()
rhosq=results.get_general_statistics()['Rho-square for the init. model'][0]
rhobsq=results.get_general_statistics()['Rho-square-bar for the init. model'][0]

print('Biogeme Results')
print(results.get_estimated_parameters())
print(f'Rhosq: {rhosq:.4f}')
print(f'RhoBarsq: {rhobsq:.4f}')

'''
Test Conclusions:
    It seems Pytorch does not normalize the error term’s variance to pisq/6
    Pytorch's CELoss using indexed y input with raw path attrs fed works almost good as biogeme in terms of MRS
    Deliberate preprocessing to input difference matrix (J-1 alt) can be efficient
'''