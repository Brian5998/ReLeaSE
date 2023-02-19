import sys
import pickle
import torch
import numpy as np
import selfies as sf 
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors, rdMMPA, QED, RDConfig, Draw, PropertyMol



class DILIPredictor(object):
    def __init__(self, path, tokens):
        self.model = pickle.load(open(path, 'rb'))
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

    def predict(self, selfies,drug, use_tqdm=False):
    
        toselfiesdict = {
        "a":'[#Branch1]',
        "b":"[#Branch2]",
        "c":'[#C]',
        "d":"[#N]",
        "e":"[=Branch1]",
        "f":'[=Branch2]',
        "g":"[=C]",
        "h":"[=N]",
        "i":'[=O]',
        "j":"[=P]",
        "k":"[=Ring1]",
        "l":'[=S]',
        "m":"[Br]",
        "n":"[Branch1]",
        "o":'[Branch2]', 
        "p":'[C]',
        "q":"[Cl]",
        "r":'[F]',
        "s":"[I]",
        "t":"[NH1]",
        "u":'[N]',
        "v":"[O]",
        "w":"[PH1]",
        "x":'[P]', 
        "y":"[Ring1]",
        "z":'[Ring2]',
        "1":"[SH1]",
        "2":"[S]",
        "3":'[nop]',
        '4':'[C@@]',
        '5':'[C@@H1]',
        "6":'[#S1]',
        "7":"[#PH1]",
        "8":"[=NH1]",
        "9":'[#SH1]',
        '=':'[#P]',
        '-':'[#SH1]',
        ')': '[#PH1]',
        '(': '[#S]',
        '<': '',
        '>': ''}
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
        preds = self.model.predict_proba(fp_smiles_list)
        predictions = []
        for arr in preds:
            predictions.append(arr[1])

        return valid_smiles, predictions, invalid_smiles
