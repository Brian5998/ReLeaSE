import sys
import pickle
import torch
import numpy as np
import selfies as sf 
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors, rdMMPA, QED, RDConfig, Draw, PropertyMol
import lightgbm as lgb
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold 
import math
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from sklearn import metrics
from scipy import stats as stats



class HemToxDeltaPredictor(object):
    def __init__(self, path, tokens):
        self.model = pickle.load(open(path, 'rb'))
        self.tokens = tokens

  
    def predict(self, selfies,drug, use_tqdm=False):
    
#         toselfiesdict = {
#         "a":'[#Branch1]',
#         "b":"[#Branch2]",
#         "c":'[#C]',
#         "d":"[#N]",
#         "e":"[=Branch1]",
#         "f":'[=Branch2]',
#         "g":"[=C]",
#         "h":"[=N]",
#         "i":'[=O]',
#         "j":"[=P]",
#         "k":"[=Ring1]",
#         "l":'[=S]',
#         "m":"[Br]",
#         "n":"[Branch1]",
#         "o":'[Branch2]', 
#         "p":'[C]',
#         "q":"[Cl]",
#         "r":'[F]',
#         "s":"[I]",
#         "t":"[NH1]",
#         "u":'[N]',
#         "v":"[O]",
#         "w":"[PH1]",
#         "x":'[P]', 
#         "y":"[Ring1]",
#         "z":'[Ring2]',
#         "1":"[SH1]",
#         "2":"[S]",
#         "3":'[nop]',
#         '4':'[C@@]',
#         '5':'[C@@H1]',
#         "6":'[#S1]',
#         "7":"[#PH1]",
#         "8":"[=NH1]",
#         "9":'[#SH1]',
#         '=':'[#P]',
#         '-':'[#SH1]',
#         ')': '[#PH1]',
#         '(': '[#S]',
#         '<': '',
#         '>': ''}
#Desalted Input ALL FRAGS
        toselfiesdict = {
          "a":'[#Branch1]',
          "b":"[#Branch2]",
          "c":'[#C]',
          "d":"[#N]",
          'e':'[/C@@H1]',
          'f':'[/C@H1]',
          'g':'[/C]',
          'h':'[/N]',
          'i':'[/O]',
          'j':'[/S]',
          'k':'[2H]',
          "l":"[=Branch1]",
          "m":'[=Branch2]',
          "n":"[=C]",
          'o':'[=N+1]',
          'p':'[=N-1]',
          "q":"[=N]",
          "r":'[=O]',
          "s":"[=P]",
          "t":"[=Ring1]",
          'u':'[=Ring2]',
          "v":'[=S]',
          'w':'[B]',
          'x':'[Br-1]',
          "y":"[Br]",
          "z":"[Branch1]",
          "A":'[Branch2]', 
          'B':'[Branch3]',
          'C':'[C@@H1]',
          'D':'[C@@]',
          'E':'[C@H1]',
          'F':'[C@]',
          "G":'[C]',
          "H":"[Cl]",
          "I":'[F]',
          "J":"[I]",
          'K':'[N+1]',
          'L':'[N-1]',
          "M":"[NH1]",
          "N":'[N]',
          'O':'[O-1]',
          "P":"[O]",
          'Q':'[P+1]',
          "R":'[P]', 
          "S":"[Ring1]",
          "T":'[Ring2]',
          'U':'[Ring3]',
          'V':'[S+1]',
          "W":"[S]",
          'X':'[Se]',
          'Y':'[Si]',
          'Z':'[\\C@@H1]',
          '1':'[\\C]',
          '2':'[\\N]',
          '3':'[\\O]',
          '4':'[\\S]',
          '<':'',
          '>':''
        }
        canonical_smiles = []
        invalid_smiles = []
        total_molecules = 0
        molecules = []
        valid_smiles = []
        API = []
        API_smiles = sf.decoder(drug)
        for selfie in selfies:
            total_molecules = total_molecules + 1
            molecule = []
            for char in selfie:
                molecule.append(toselfiesdict[char])
            real_selfie = drug + ''.join(molecule)
            smiles = sf.decoder(real_selfie)
            if Chem.MolFromSmiles(smiles) is not None:
                valid_smiles.append(smiles)
                API.append(API_smiles)
            else: 
                invalid_smiles.append(smiles)
        ##
        lip = pd.DataFrame()
        lip['API SMILES'] = API
        lip['Prodrug SMILES'] = valid_smiles
        APIs = [Chem.MolFromSmiles(s) for s in lip["API SMILES"]]
        APIfps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in APIs]
        Pros = [Chem.MolFromSmiles(s) for s in lip["Prodrug SMILES"]]
        Profps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in Pros]
        Data = pd.DataFrame(data={'API': list(np.array(APIfps).astype(bool)), 'Pros': list(np.array(Profps).astype(bool))})
        del APIs, APIfps, Pros, Profps
        Data["Fingerprint"] = Data.Pros.combine(Data.API, np.append) # concatenate ExplicitBitVec objects from RDKIT
        Data = Data.drop(["API", "Pros"], axis=1)
        if(Data.size > 0):   
            delta = self.model.predict(np.vstack(Data.Fingerprint.to_numpy()))
            preds = delta.tolist()
        else:
            preds = -1.0
        
        return valid_smiles, preds, invalid_smiles
