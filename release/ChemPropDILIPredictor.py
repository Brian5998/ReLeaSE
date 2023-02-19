import sys
import torch
import numpy as np
import selfies as sf 
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors, rdMMPA, QED, RDConfig, Draw, PropertyMol
import chemprop
import pandas as pd



class ChemPropDILIPredictor(object):
    def __init__(self, path, tokens):
      
        self.path = path
        self.tokens = tokens
        self.model_arguments = [
    '--checkpoint_dir', self.path,
    '--test_path','Temp.csv',
    '--preds_path', None,
    '--number_of_molecules', '1'
    ]
        self.args = chemprop.args.PredictArgs().parse_args(self.model_arguments)

        self.model_objects = chemprop.train.load_model(args=self.args)
    
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
        to_pred = pd.DataFrame(valid_smiles, columns = ['Drug'])
        to_pred.to_csv('Temp.csv',index=False)
        
    #predict them and save predictions 
        preds =chemprop.train.make_predictions(
            args=self.args, model_objects=self.model_objects)
        if len(preds) > 0:
            preds = list(np.concatenate(preds).flat)

        return valid_smiles, preds, invalid_smiles

    def predictFromSmiles(self, smiles, use_tqdm=False):
        invalid_smiles = []
        to_pred = pd.DataFrame(smiles, columns = ['Drug'])
        to_pred.to_csv('Temp.csv',index=False)
        
    #predict them and save predictions 
        preds =chemprop.train.make_predictions(
            args=self.args, model_objects=self.model_objects)
        if len(preds) > 0:
            preds = list(np.concatenate(preds).flat)

        return smiles, preds, invalid_smiles