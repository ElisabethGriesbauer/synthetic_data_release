"""
Procedures for running a privacy evaluation on a generative model
"""

from numpy import where, mean, std

import json
from os import path
from utils.utils import json_numpy_serialzer
from math import sqrt
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame

from utils.constants import *

def get_accuracy(guesses, labels, targetPresence):
    idxIn = where(targetPresence == LABEL_IN)[0]
    idxOut = where(targetPresence == LABEL_OUT)[0]

    pIn = sum([g == l for g,l in zip(guesses[idxIn], labels[idxIn])])/len(idxIn)
    pOut = sum([g == l for g,l in zip(guesses[idxOut], labels[idxOut])])/len(idxOut)
    return pIn, pOut


def get_tp_fp_rates(guesses, labels):
    targetIn = where(labels == LABEL_IN)[0]
    targetOut = where(labels == LABEL_OUT)[0]
    return sum(guesses[targetIn] == LABEL_IN)/len(targetIn), sum(guesses[targetOut] == LABEL_IN)/len(targetOut)


def get_probs_correct(pdf, targetPresence):
    idxIn = where(targetPresence == LABEL_IN)[0]
    idxOut = where(targetPresence == LABEL_OUT)[0]

    # pdf[pdf > 1.] = 1.
    return mean(pdf[idxIn]), mean(pdf[idxOut])


def get_mia_advantage(tp_rate, fp_rate):
    return tp_rate - fp_rate


def get_ai_advantage(pCorrectIn, pCorrectOut):
    return pCorrectIn - pCorrectOut


def get_ai_odds(pCorrectS, pCorrectR):
    if pCorrectR ==0:
        odds = float('nan')
    else:
        odds = pCorrectS/pCorrectR
    return odds


def get_util_advantage(pCorrectIn, pCorrectOut):
    return pCorrectIn - pCorrectOut


def get_prob_removed(before, after):
    idxIn = where(before == LABEL_IN)[0]
    return 1.0 - sum(after[idxIn]/len(idxIn))

def remove_empty_dicts(d):
    if isinstance(d, dict):
        keys_to_remove = []
        for key, value in d.items():
            if isinstance(value, dict):
                remove_empty_dicts(value)  # Recursively check nested dictionaries
                if not value:  # If the nested dictionary is now empty, mark the key for removal
                    keys_to_remove.append(key)
            elif not value:  # If the value is empty (None, empty dict, etc.), mark the key for removal
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            d.pop(key)  # Remove the marked keys from the dictionary

def standardize_before_AIA(data, metadata, scaler):
    cols_to_exclude = [col['name'] for col in metadata['columns'] if col['type'] in ['Categorical', 'Ordinal']]
    cols_to_scale = [col for col in data.columns if col not in cols_to_exclude]
    col_indices = [data.columns.get_loc(col) for col in cols_to_scale]

    # Fit and transform only the selected columns
    scaled_values = scaler.fit_transform(data.iloc[:, col_indices])

    # Create a new DataFrame with the scaled values
    standBootstrapSample = DataFrame(data=scaled_values, columns=cols_to_scale)
    standBootstrapSample.index = data.index

    # Add the excluded columns back to the standardized DataFrame
    for col in cols_to_exclude:
        standBootstrapSample[col] = data[col]
    
    return standBootstrapSample


from utils.logging import LOGGER

class Evaluator():
    
    def __init__(self, rawTout, sizeSynT, nSynT, attacks, targetIDs, alpha, resultsTargetPrivacy, nr, targets, standTargets, args):
        self.rawTout = rawTout
        self.sizeSynT = sizeSynT
        self. nSynT = nSynT
        self.attacks = attacks
        self.targetIDs = targetIDs
        self.alpha = alpha
        self.resultsTargetPrivacy = resultsTargetPrivacy
        self.nr = nr
        self.targets = targets
        self.standTargets = standTargets
        self.args = args

    def fit_and_evaluate(self, GenModel):
        LOGGER.info(f'Start: Evaluation for model {GenModel.__name__}...')

        D = self.rawTout.shape[1] - 1
        GenModel.fit(self.rawTout)
        synTwithoutTarget = [GenModel.generate_samples(self.sizeSynT) for _ in range(self.nSynT)]
        for sa, Attack in self.attacks.items():
            for tid in self.targetIDs:
                self.resultsTargetPrivacy[tid][sa][GenModel.__name__][self.nr] = {
                    'AttackerGuess': [],
                    'ProbCorrect': [],
                    'MSE': [],
                    'TargetPresence': [LABEL_OUT for _ in range(self.nSynT)]
                }
            for syn in synTwithoutTarget:
                scaler = StandardScaler()
                standSyn = standardize_before_AIA(syn, metadata, scaler)
                assert not standSyn.isna().any().any()   
                # scaler.fit(syn.drop('Y', axis = 1))
                # standSyn = syn
                # standSyn.iloc[:,:D] = DataFrame(scaler.transform(syn.drop('Y', axis = 1)))
                Attack.train(standSyn)
                
                for tid in self.targetIDs:
                    target = self.standTargets.loc[[tid]]
                    targetAux = target.loc[[tid], Attack.knownAttributes]
                    targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                    guess = Attack.attack(targetAux)

                    # don't need to average over nSynT here because this is done in load_results_inference
                    pCorrect = int( targetSecret - self.alpha < guess < targetSecret + self.alpha )
                    mse = (guess - targetSecret)**2
                    
                    self.resultsTargetPrivacy[tid][sa][GenModel.__name__][self.nr]['AttackerGuess'].append(guess)
                    self.resultsTargetPrivacy[tid][sa][GenModel.__name__][self.nr]['ProbCorrect'].append(pCorrect)
                    self.resultsTargetPrivacy[tid][sa][GenModel.__name__][self.nr]['MSE'].append(mse)
        
        del synTwithoutTarget
                
        for tid in self.targetIDs:
            LOGGER.info(f'Target: {tid}')
            target = self.targets.loc[[tid]]
            rawTin = self.rawTout.append(target)
            rawTin['Y'] = rawTin['Y'].astype(str)
            GenModel.fit(rawTin)
            synTwithTarget = [GenModel.generate_samples(self.sizeSynT) for _ in range(self.nSynT)]

            # for regression
            target = self.standTargets.loc[[tid]]

            for sa, Attack in self.attacks.items():
                targetAux = target.loc[[tid], Attack.knownAttributes]
                targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                for syn in synTwithTarget:
                    standSyn = standardize_before_AIA(syn, metadata, scaler)
                    assert  not standSyn.isna().any().any()
                    # scaler.fit(syn.drop('Y', axis = 1))
                    # standSyn = syn
                    # standSyn.iloc[:,:D] = DataFrame(scaler.transform(syn.drop('Y', axis = 1)))
                    Attack.train(standSyn)

                    guess = Attack.attack(targetAux)
                    
                    # don't need to average over nSynT here because this is done in load_results_inference
                    pCorrect = int( targetSecret - self.alpha < guess < targetSecret + self.alpha )
                    mse = (guess - targetSecret)**2

                    self.resultsTargetPrivacy[tid][sa][GenModel.__name__][self.nr]['AttackerGuess'].append(guess)
                    self.resultsTargetPrivacy[tid][sa][GenModel.__name__][self.nr]['ProbCorrect'].append(pCorrect)
                    self.resultsTargetPrivacy[tid][sa][GenModel.__name__][self.nr]['MSE'].append(mse)
                    self.resultsTargetPrivacy[tid][sa][GenModel.__name__][self.nr]['TargetPresence'].append(LABEL_IN)
        del synTwithTarget


        remove_empty_dicts(self.resultsTargetPrivacy)

        outfile = f"ResultsMLEAI_{GenModel.__name__}_run_{self.nr}"

        with open(path.join(f'{self.args.outdir}', f'{outfile}.json'), 'w') as f:
            json.dump(self.resultsTargetPrivacy, f, indent=2, default=json_numpy_serialzer)

        return self.resultsTargetPrivacy

