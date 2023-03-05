import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
import selfies as sf
from statistics import mean

import numpy
from matplotlib.ticker import PercentFormatter


class SimilarityCalculator(object):
    def __init__(self, tokens):
        self.tokens = tokens

    def predict(self, selfies, drug, use_tqdm=False):
    
        #CHEMBL Input
        toselfiesdict = {
          "a":'[#Branch1]',
          "b":"[#Branch2]",
          'c':'[#C-1]',
          "d":'[#C]',
          'e':'[#N+1]',
          'f':'[#N]',
          'g':'[#O+1]',
          'h':'[-/Ring2]',
          'i':'[-\\Ring1]',
          'j':'[-\\Ring2]',
          'k':'[/C]',
          'l':'[/N]',
          'm':'[/O]',
          'n':'[/S]',
          'o':'[=Branch1]',
          "p":"[=Branch2]",
          "q":'[=Branch3]',
          "r":"[=CH1]",
          "s":"[=C]",
          't':'[=I+2]',
          'u':'[=N+1]',
          'v':'[=N-1]',
          'w':'[=NH1+1]',
          'x':'[=NH2+1]',
          "y":"[=N]",
          'z':'[=O+1]',
          'A':'[=OH1+1]',
          "B":'[=O]',
          'C':'[=P+1]',
          "D":"[=P]",
          "E":"[=Ring1]",
          'F':'[=Ring2]',
          'G':'[=Ring3]',
          'H':'[=S+1]',
          "I":'[=S]',
          'J':'[B-1]',
          'K':'[B]',
          "L":"[Br]",
          "M":"[Branch1]",
          "N":'[Branch2]', 
          'O':'[Branch3]',
          'P':'[C+1]',
          'Q':'[C-1]',
          'R':'[CH0]',
          'S':'[CH1-1]',
          'T':'[CH1]',
          'U':'[CH2]',
          'V':'[C]',
          'W':'[Cl]',
          'X':'[F]',
          'Y':'[I+1]',
          'Z':'[I]',
          '1':'[N+1]',
          '2':'[N-1]',
          '3':'[NH0]',
          '4':'[NH1+1]',
          '5':'[NH1-1]',
          '6':'[NH1]',
          '7':'[NH2+1]',
          '8':'[NH3+1]',
          '9':'[N]',
          '0':'[O+1]',
          '!':'[O-1]',
          '@':'[OH0]',
          '#':'[O]',
          '$':'[P+1]',
          '%':'[PH0]',
          '^':'[P]',
          '&':'[Ring1]',
          '*':'[Ring2]',
          '(':'[Ring3]',
          ')':'[S+1]',
          ':':'[S-1]',
          ';':'[S]',
          '{':'[\\C]',
          '}':'[\\N]',
          '`':'[\\O]',
          '<':'',
          '>':''
         }

# #Desalted Input ALL FRAGS
#         toselfiesdict = {
#           "a":'[#Branch1]',
#           "b":"[#Branch2]",
#           "c":'[#C]',
#           "d":"[#N]",
#           'e':'[/C@@H1]',
#           'f':'[/C@H1]',
#           'g':'[/C]',
#           'h':'[/N]',
#           'i':'[/O]',
#           'j':'[/S]',
#           'k':'[2H]',
#           "l":"[=Branch1]",
#           "m":'[=Branch2]',
#           "n":"[=C]",
#           'o':'[=N+1]',
#           'p':'[=N-1]',
#           "q":"[=N]",
#           "r":'[=O]',
#           "s":"[=P]",
#           "t":"[=Ring1]",
#           'u':'[=Ring2]',
#           "v":'[=S]',
#           'w':'[B]',
#           'x':'[Br-1]',
#           "y":"[Br]",
#           "z":"[Branch1]",
#           "A":'[Branch2]', 
#           'B':'[Branch3]',
#           'C':'[C@@H1]',
#           'D':'[C@@]',
#           'E':'[C@H1]',
#           'F':'[C@]',
#           "G":'[C]',
#           "H":"[Cl]",
#           "I":'[F]',
#           "J":"[I]",
#           'K':'[N+1]',
#           'L':'[N-1]',
#           "M":"[NH1]",
#           "N":'[N]',
#           'O':'[O-1]',
#           "P":"[O]",
#           'Q':'[P+1]',
#           "R":'[P]', 
#           "S":"[Ring1]",
#           "T":'[Ring2]',
#           'U':'[Ring3]',
#           'V':'[S+1]',
#           "W":"[S]",
#           'X':'[Se]',
#           'Y':'[Si]',
#           'Z':'[\\C@@H1]',
#           '1':'[\\C]',
#           '2':'[\\N]',
#           '3':'[\\O]',
#           '4':'[\\S]',
#           '<':'',
#           '>':''
#         }
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
        
        Generated = [Chem.MolFromSmiles(mol) for mol in valid_smiles]
        Generated_fps_list = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in Generated]
        
        explosive_list = ['[N-]=[N+]=N/C(N=[N+]=[N-])=N\\N1N=NN=C1N=[N+]=[N-]']
        
        #explosive_list = [''N1(COOC2)COOCN2COOC1','CC1(C)OOC(C)(C)OOC(C)(C)OO1','[N-]=[N+]=N/C(N=[N+]=[N-])=N\\N1N=NN=C1N=[N+]=[N-]','O=[N+]([O-])N1CN([N+]([O-])=O)CN([N+]([O-])=O)C1','N1(COOC2)COOCN2COOC1','CC1=C([N+]([O-])=O)C=C([N+]([O-])=O)C=C1[N+]([O-])=O','O=[N+]([O-])OCC(CO[N+]([O-])=O)(CO[N+]([O-])=O)CO[N+]([O-])=O','O=[N+]([O-])OCC(O[N+]([O-])=O)CO[N+]([O-])=O']
        Explosive = [Chem.MolFromSmiles(mol) for mol in explosive_list]
        Explosive_fps_list = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in Explosive]
        
        similarity_list = []
        for i in range(len(Generated_fps_list)):
            single_sim = []
            for j in range(len(Explosive_fps_list)):
                single_sim.append(DataStructs.TanimotoSimilarity(Explosive_fps_list[j], Generated_fps_list[i]))
            similarity_list.append(single_sim)
            if single_sim == 1.0:
                print('Bomb was made') 
        
        predictions = [max(predlist) for predlist in similarity_list]
        return valid_smiles, predictions, invalid_smiles

    
    
    
    
    
    
    
    
    
    def predictFromSmiles(self, smiles, use_tqdm=False):
        invalid_smiles = []
        
        Generated = [Chem.MolFromSmiles(mol) for mol in smiles]
        Generated_fps_list = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in Generated]
        
        explosive_list = ['[N-]=[N+]=N/C(N=[N+]=[N-])=N\\N1N=NN=C1N=[N+]=[N-]']
        Explosive = [Chem.MolFromSmiles(mol) for mol in explosive_list]
        Explosive_fps_list = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in Explosive]
        
        similarity_list = []
        for i in range(len(Generated_fps_list)):
            single_sim = []
            for j in range(len(Explosive_fps_list)):
                single_sim.append(DataStructs.TanimotoSimilarity(Explosive_fps_list[j], Generated_fps_list[i]))
            similarity_list.append(single_sim)
        
        preds = [max(predlist) for predlist in similarity_list]

        return smiles, preds, invalid_smiles
