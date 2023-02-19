import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
import selfies as sf

import numpy
from matplotlib.ticker import PercentFormatter


class SimilarityCalculatorSMILES(object):
    def __init__(self, tokens):
        self.tokens = tokens

    def predict(self, smiles, drug, use_tqdm=False):
    
        canonical_smiles = []
        invalid_smiles = []
        total_molecules = 0
        molecules = []
        valid_smiles = []
        for smile in smiles:
            if Chem.MolFromSmiles(smile) is not None:
                valid_smiles.append(smile)
            else: 
                invalid_smiles.append(smile)
        
        Generated = [Chem.MolFromSmiles(mol) for mol in valid_smiles]
        Generated_fps_list = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in Generated]
        
        explosive_list = ['CC1(C)OOC(C)(C)OOC(C)(C)OO1','[N-]=[N+]=N/C(N=[N+]=[N-])=N\\N1N=NN=C1N=[N+]=[N-]','O=[N+]([O-])N1CN([N+]([O-])=O)CN([N+]([O-])=O)C1','N1(COOC2)COOCN2COOC1','CC1=C([N+]([O-])=O)C=C([N+]([O-])=O)C=C1[N+]([O-])=O','O=[N+]([O-])OCC(CO[N+]([O-])=O)(CO[N+]([O-])=O)CO[N+]([O-])=O','O=[N+]([O-])OCC(O[N+]([O-])=O)CO[N+]([O-])=O']
        Explosive = [Chem.MolFromSmiles(mol) for mol in explosive_list]
        Explosive_fps_list = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in Explosive]
        
        similarity_list = []
        for i in range(len(Generated_fps_list)):
            single_sim = []
            for j in range(len(Explosive_fps_list)):
                single_sim.append(DataStructs.TanimotoSimilarity(Explosive_fps_list[j], Generated_fps_list[i]))
            similarity_list.append(single_sim)
        
        predictions = [max(predlist) for predlist in similarity_list]
        return valid_smiles, predictions, invalid_smiles

    def predictFromSmiles(self, smiles, use_tqdm=False):
        invalid_smiles = []
        
        Generated = [Chem.MolFromSmiles(mol) for mol in smiles]
        Generated_fps_list = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in Generated]
        
        explosive_list = ['CC1(C)OOC(C)(C)OOC(C)(C)OO1','[N-]=[N+]=N/C(N=[N+]=[N-])=N\\N1N=NN=C1N=[N+]=[N-]','O=[N+]([O-])N1CN([N+]([O-])=O)CN([N+]([O-])=O)C1','N1(COOC2)COOCN2COOC1','CC1=C([N+]([O-])=O)C=C([N+]([O-])=O)C=C1[N+]([O-])=O','O=[N+]([O-])OCC(CO[N+]([O-])=O)(CO[N+]([O-])=O)CO[N+]([O-])=O','O=[N+]([O-])OCC(O[N+]([O-])=O)CO[N+]([O-])=O']
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