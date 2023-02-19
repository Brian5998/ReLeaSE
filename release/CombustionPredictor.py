import sys
import pickle
import torch
import numpy as np
import selfies as sf 
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors, rdMMPA, QED, RDConfig, Draw, PropertyMol



class CombustionPredictor(object):
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
        if(len(fp_smiles_list) > 0):
            preds = self.model.predict(fp_smiles_list)
        else:
            preds = 0

        return valid_smiles, preds, invalid_smiles
