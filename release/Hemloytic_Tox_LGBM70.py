import sys
import pickle
import torch
import numpy as np
import selfies as sf 
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors, rdMMPA, QED, RDConfig, Draw, PropertyMol
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import train_test_split



class HemToxPredictor(object):
    def __init__(self, path, tokens):
        with open(path, 'rb') as file:
            self.model = pickle.load(file)
        self.tokens = tokens

    def morgan_fp(self, mol, radius=3, nbits=2048, use_features=False):
        "morgan fingerprint"
        mol = self.to_mol(mol)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, useFeatures=use_features)
        return fp


    def fp_to_array(self, fp):
        "Converts RDKit `ExplicitBitVec` to numpy array"
        return np.unpackbits(np.frombuffer(DataStructs.BitVectToBinaryText(fp), dtype=np.uint8), bitorder='little')
    
            
            
            
    def to_mol(self, smile_or_mol):
        if (type(smile_or_mol) == str) or (type(smile_or_mol) == np.str_):
            mol = Chem.MolFromSmiles(smile_or_mol)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except:
                    mol = None
        else:
            mol = smile_or_mol

        return mol

    def dofit(self):
        data = pd.read_csv('/hpc/group/rekerlab/byz6/ReLeaSE/release/HemolyticToxicityCorrect.csv')
        X = data['SMILES'].tolist()
        Y = data['Y'].tolist()
        molecules = []
        for mol in X:
            molecules.append(self.morgan_fp(mol))

        X_data = []
        for arr in molecules:
            X_data.append(self.fp_to_array(arr))
        X_ndarray = np.array(X_data)
        y_ndarray = np.array(data['Y'].tolist())
        X_ndarray_train,X_ndarray_test,y_ndarray_train,y_ndarray_test = train_test_split(X_ndarray,
                                                 y_ndarray,
                                                 test_size=0.30,
                                                 random_state=42)
        model = self.model
        model.fit(X_ndarray_train,y_ndarray_train)
        
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
        for selfie in selfies:
            total_molecules = total_molecules + 1
            molecule = []
            for char in selfie:
                molecule.append(toselfiesdict[char])
            real_selfie = drug + ''.join(molecule)
            smiles = sf.decoder(real_selfie)
            if Chem.MolFromSmiles(smiles) is not None:
                valid_smiles.append(smiles)
            else: 
                invalid_smiles.append(smiles)
        for mol in valid_smiles:
            molecules.append(self.morgan_fp(mol))
        
        fp_smiles_list = []
        for arr in molecules:
            fp_smiles_list.append(self.fp_to_array(arr))

    #predict them and save predictions 
        
        preds = self.model.predict(fp_smiles_list)
        predictions = []
        for arr in preds:
            predictions.append(arr[1])

        return valid_smiles, predictions, invalid_smiles
