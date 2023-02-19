import torch
print(torch.cuda.is_available())
import sys
sys.path.append('./release/')
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()
import numpy as np
from tqdm import tqdm, trange
import pickle
from rdkit import Chem, DataStructs
from stackRNN import StackAugmentedRNN
from data import GeneratorData 
from utils import canonical_smiles
import matplotlib.pyplot as plt

import seaborn as sns


gen_data_path = './data/Azidoazide_Tailored.smi'#TAILORED
tokens = ['<', '>','a','b','c','d','k','m','o','f','p','q','s','u','v','y','B','E','F','G','Q','V','M','N','O','1','2','9','#','@','}','`','&','s','*']
gen_data = GeneratorData(training_data_path=gen_data_path, delimiter=',', 
                         cols_to_read=[0], keep_header=True, tokens=tokens)
                         
def plot_hist(prediction, n_to_generate):
    prediction = np.array(prediction)
    percentage_in_threshold = np.sum((prediction >= 0.5) & 
                                     (prediction <= 1.0))/len(prediction)
    print("Percentage of predictions within similarity region:", percentage_in_threshold)
    print("Proportion of valid SMILES:", len(prediction)/n_to_generate)
    ax = sns.kdeplot(prediction, shade=True)
    plt.axvline(x=0.5)
    plt.axvline(x=1.0)
    ax.set(xlabel='Predicted Similarity', 
           title='Distribution of predicted Similarity for generated molecules')
    plt.show()
    
def estimate_and_update(generator, predictor, drug, n_to_generate):
    generated = []
    pbar = tqdm(range(n_to_generate))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        generated.append(generator.evaluate(gen_data, predict_len=150)[1:-1])

#     sanitized = canonical_smiles(generated, sanitize=False, throw_warning=False)[:-1]
#     unique_smiles = list(np.unique(sanitized))[1:]
    smiles, prediction, nan_smiles = predictor.predict(generated, drug, use_tqdm=True)  
    print(len(prediction))
                                                       
    plot_hist(prediction, n_to_generate)
        
    return smiles, prediction

hidden_size = 1500
stack_width = 1500
stack_depth = 200
layer_type = 'GRU'
lr = 0.001
optimizer_instance = torch.optim.Adadelta

my_generator = StackAugmentedRNN(input_size=gen_data.n_characters, hidden_size=hidden_size,
                                 output_size=gen_data.n_characters, layer_type=layer_type,
                                 n_layers=1, is_bidirectional=False, has_stack=True,
                                 stack_width=stack_width, stack_depth=stack_depth, 
                                 use_cuda=use_cuda, 
                                 optimizer_instance=optimizer_instance, lr=lr)

model_path = './trainedModels/generator/AzidoazideTailored'

losses = my_generator.fit(gen_data,1000)
plt.plot(losses)
my_generator.evaluate(gen_data)
my_generator.save_model(model_path)
my_generator.load_model(model_path)
#tanimoto sim 
from SimilarityCalculator import SimilarityCalculator
my_predictor = SimilarityCalculator(tokens)

selfies = ['VV','VVMVV##VMVVMVV##VMVVMVV##&s','asdasuznx']
drug = ''
mol, pred, invalid = my_predictor.predict(selfies, drug)

smiles_unbiased, prediction_unbiased = estimate_and_update(my_generator,
                                                           my_predictor,
                                                           drug,
                                                           n_to_generate=1000)
                                                           
from reinforcement import Reinforcement


my_generator_max = StackAugmentedRNN(input_size=gen_data.n_characters, 
                                     hidden_size=hidden_size,
                                     output_size=gen_data.n_characters, 
                                     layer_type=layer_type,
                                     n_layers=1, is_bidirectional=False, has_stack=True,
                                     stack_width=stack_width, stack_depth=stack_depth, 
                                     use_cuda=use_cuda, 
                                     optimizer_instance=optimizer_instance, lr=lr)

my_generator_max.load_model(model_path)


# Setting up some parameters for the experiment
n_to_generate = 200
n_policy_replay = 10
n_policy = 20
n_iterations = 200

def simple_moving_average(previous_values, new_value, ma_window_size=10):
    value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
    value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
    return value_ma
    
def get_reward_logp(smiles, predictor, invalid_reward=0.0):
    drug = ''
    mol, prop, nan_smiles = predictor.predict([smiles],drug)
    if len(nan_smiles) == 1:
        return invalid_reward
#     if (prop[0] < 0.7):
    else:
        return 10*(np.tanh((3*prop[0]) - 1))+5
#     else:
#         return 1000


x = np.linspace(0, 1)
reward = lambda x: 10*(np.tanh((3*x)-1))+5 if x > -10 else 1000
plt.plot(x, [reward(i) for i in x])
plt.xlabel('Similarity value')
plt.ylabel('Reward value')
plt.title('Reward function for Similarity optimization')
plt.show()


RL_logp = Reinforcement(my_generator_max, my_predictor, get_reward_logp)

rewards = []
rl_losses = []

for i in range(n_iterations):
    for j in trange(n_policy, desc='Policy gradient...'):
        cur_reward, cur_loss = RL_logp.policy_gradient(gen_data)
        rewards.append(simple_moving_average(rewards, cur_reward)) 
        rl_losses.append(simple_moving_average(rl_losses, cur_loss))
    
    plt.plot(rewards)
    plt.xlabel('Training iteration')
    plt.ylabel('Average reward')
    plt.show()
    plt.plot(rl_losses)
    plt.xlabel('Training iteration')
    plt.ylabel('Loss')
    plt.show()
        
    smiles_cur, prediction_cur = estimate_and_update(RL_logp.generator, 
                                                     my_predictor, drug, 
                                                     n_to_generate)
    print('Sample trajectories:')
    for sm in smiles_cur[:5]:
        print(sm)
    print(str(i+1) + " out of " + str(n_iterations) + " complete!")
    
smiles_biased, prediction_biased = estimate_and_update(RL_logp.generator, 
                                                       my_predictor,
                                                       drug,
                                                       n_to_generate=1000)

sns.kdeplot(prediction_biased, label='Optimized', shade=True, color='purple')
sns.kdeplot(prediction_unbiased, label='Unbiased', shade=True, color='grey')
plt.xlabel('Similarity value')
plt.title('Initial and biased distributions of Similarity')
plt.legend()
plt.show()

