"""
Command-line interface for running privacy evaluation under an attribute inference adversary
"""

import json

from os import mkdir, path
from numpy.random import choice, seed, randint
from numpy import mean, std, append
from argparse import ArgumentParser
# for checking regression coefficients
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame

from utils.datagen import load_s3_data_as_df, load_local_data_as_df
from utils.utils import json_numpy_serialzer
from utils.logging import LOGGER
from utils.constants import *

from generative_models.ctganSDV import CTGAN
from generative_models.tvae import TVAE
# from generative_models.tvae import TVAE
from generative_models.data_synthesiser import (IndependentHistogram,
                                                BayesianNet,
                                                PrivBayes, #added:
                                                Rvine,
                                                Cvine,
                                                Rvinestar1)
from generative_models.pate_gan import PATEGAN
from sanitisation_techniques.sanitiser import SanitiserNHS
# from attack_models.reconstruction import LinRegAttack, RandForestAttack
from attack_models.reconstruction import LinRegAttack, RandForestAttack
from utils.evaluation_framework import standardize_before_AIA

from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)
simplefilter('ignore', category=DeprecationWarning)

cwd = path.dirname(__file__)

SEED = 42


def main():
    argparser = ArgumentParser()
    datasource = argparser.add_mutually_exclusive_group()
    datasource.add_argument('--s3name', '-S3', type=str, choices=['adult', 'census', 'credit', 'alarm', 'insurance'], help='Name of the dataset to run on')
    datasource.add_argument('--datapath', '-D', type=str, help='Relative path to cwd of a local data file')
    argparser.add_argument('--runconfig', '-RC', default='runconfig_mia.json', type=str, help='Path relative to cwd of runconfig file')
    argparser.add_argument('--outdir', '-O', default='tests', type=str, help='Path relative to cwd for storing output files')
    argparser.add_argument('--coeff', '-C', type=int, default = 1, help='save regression coefficients: yes = 1, no = 0')
    args = argparser.parse_args()

    # Load runconfig
    with open(path.join(cwd, args.runconfig)) as f:
        runconfig = json.load(f)
    print('Runconfig:')
    print(runconfig)

    # Load data
    if args.s3name is not None:
        rawPop, metadata = load_s3_data_as_df(args.s3name)
        dname = args.s3name
    else:
        rawPop, metadata = load_local_data_as_df(path.join(cwd, args.datapath))
        dname = args.datapath.split('/')[-1]

    print(f'Loaded data {dname}:')
    print(rawPop.info())

    # Make sure outdir exists
    if not path.isdir(args.outdir):
        mkdir(args.outdir)

    seed(SEED)

    ########################
    #### GAME INPUTS #######
    ########################
    D = rawPop.shape[1] - 1
    # Standardize rawPop
    scaler = StandardScaler()
    standRawPop = standardize_before_AIA(rawPop, metadata, scaler)
    # standRawPop = rawPop.copy()
    # scaler.fit(standRawPop.drop('Y', axis = 1))
    # standRawPop.iloc[:,:D] = DataFrame(scaler.transform(standRawPop.drop('Y', axis = 1)), index = rawPop.index.values)
    
    # Pick targets
    targetIDs = choice(list(rawPop.index), size=runconfig['nTargets'], replace=False).tolist()

    # If specified: Add specific target records
    if runconfig['Targets'] is not None:
        targetIDs.extend(runconfig['Targets'])

    # Pick targets from (non-)standardized data set:
    # standardized targets are for regression fitting, non-stand targets for synth data generation
    standTargets = standRawPop.loc[targetIDs, :]
    targets = rawPop.loc[targetIDs, :]

    # Drop targets from population
    rawPopDropTargets = rawPop.drop(targetIDs)

    # List of candidate generative models to evaluate
    gmList = []
    if 'generativeModels' in runconfig.keys():
        for gm, paramsList in runconfig['generativeModels'].items():
            if gm == 'IndependentHistogram':
                for params in paramsList:
                    gmList.append(IndependentHistogram(metadata, *params))
            elif gm == 'BayesianNet':
                for params in paramsList:
                    gmList.append(BayesianNet(metadata, *params))
            elif gm == 'PrivBayes':
                for params in paramsList:
                    gmList.append(PrivBayes(metadata, *params))
            elif gm == 'CTGAN':
                for params in paramsList:
                    gmList.append(CTGAN(metadata, *params))
            elif gm == 'PATEGAN':
                for params in paramsList:
                    gmList.append(PATEGAN(metadata, *params))
            # added:
            elif gm == 'TVAE':
               for params in paramsList:
                   gmList.append(TVAE(metadata, *params))
            elif gm == 'Rvine':
                for params in paramsList:
                    gmList.append(Rvine(metadata, *params))
            elif gm == 'Rvinestar1':
                for params in paramsList:
                    gmList.append(Rvinestar1(metadata, *params))
            elif gm == 'Cvine':
                for params in paramsList:
                    gmList.append(Cvine(metadata, *params))
            else:
                raise ValueError(f'Unknown GM {gm}')
            
    # List of candidate sanitisation techniques to evaluate
    sanList = []
    if 'sanitisationTechniques' in runconfig.keys():
        for name, paramsList in runconfig['sanitisationTechniques'].items():
            if name == 'SanitiserNHS':
                for params in paramsList:
                    sanList.append(SanitiserNHS(metadata, *params))
            else:
                raise ValueError(f'Unknown sanitisation technique {name}')

    ##################################
    ######### EVALUATION #############
    ##################################
    resultsTargetPrivacy = {tid: {sa: {gm.__name__: {} for gm in gmList + sanList} for sa in runconfig['sensitiveAttributes']} for tid in targetIDs}
    # Add entry for raw
    for tid in targetIDs:
        for sa in runconfig['sensitiveAttributes']:
            resultsTargetPrivacy[tid][sa]['Raw'] = {}

    # for checking regression coefficients
    column_names = [f'b{i}' for i in range(D+1)] + ["sa", "tid", "genModel"]
    regCoeff = [] # pd.DataFrame(columns=column_names)

    print('\n---- Start the game ----')
    for nr in range(runconfig['nIter']):
        print(f'\n--- Game iteration {nr + 1} ---')
        # Draw a raw dataset
        rIdx = choice(list(rawPopDropTargets.index), size=runconfig['sizeRawT'], replace=False).tolist()
        rawTout = rawPopDropTargets.loc[rIdx]


        ###############
        ## ATTACKS ####
        ###############
        attacks = {}
        for sa, atype in runconfig['sensitiveAttributes'].items():
            if atype == 'LinReg':
                attacks[sa] = LinRegAttack(sensitiveAttribute=sa, metadata=metadata)
            elif atype == 'Classification':
                attacks[sa] = RandForestAttack(sensitiveAttribute=sa, metadata=metadata)

        for sa, Attack in attacks.items():
            for tid in targetIDs:
                resultsTargetPrivacy[tid][sa]['Raw'][nr] = {
                    'AttackerGuess': [],
                    'ProbCorrect': [],
                    'MSE': [],
                    'TargetPresence': [LABEL_OUT for _ in range(runconfig['nSynT'])]
                }
                
        #### Assess advantage raw
        for b in range(runconfig['nSynT']): # no. bootstrap samples needs to be the same as syntheti data sets generated
            bootstrap_sample = rawTout.iloc[randint(low = 0, high = rawTout.shape[0] - 1, size = runconfig["bootstrapSize"])]    
            attackedFeatures = list(attacks.items())

            standBootstrapSample = standardize_before_AIA(bootstrap_sample, metadata, scaler)
            assert not standBootstrapSample.isna().any().any()
            
            # scaler.fit(bootstrap_sample.drop('Y', axis = 1))
            # standBootstrapSample = bootstrap_sample
            # standBootstrapSample.iloc[:,:D] = DataFrame(scaler.transform(bootstrap_sample.drop('Y', axis = 1)), index = bootstrap_sample.index.values)
            # count = 0

            for sa, Attack in attacks.items():

                Attack.train(standBootstrapSample)
                
                for tid in targetIDs:
                # for i in range(len(targetIDs)):
                    target = standTargets.loc[[tid]]
                    targetAux = target.loc[[tid], Attack.knownAttributes]
                    targetSecret = target.loc[tid, Attack.sensitiveAttribute]
                    
                    guess = Attack.attack(targetAux, attemptLinkage=False, data=standBootstrapSample)
                    pCorrect = int( targetSecret - runconfig["alpha"] < guess < targetSecret + runconfig["alpha"] )
                    mse = (guess - targetSecret)**2

                    resultsTargetPrivacy[tid][sa]['Raw'][nr]['AttackerGuess'].append(guess)
                    resultsTargetPrivacy[tid][sa]['Raw'][nr]['ProbCorrect'].append(pCorrect)
                    resultsTargetPrivacy[tid][sa]['Raw'][nr]['MSE'].append(mse)
            
        del bootstrap_sample, standBootstrapSample


        for tid in targetIDs:
            # add non-standardized target to data
            target = targets.loc[[tid]]
            rawTin = rawTout.append(target)

            # for regression
            target = standTargets.loc[[tid]]

            for sa, Attack in attacks.items():
                targetAux = target.loc[[tid], Attack.knownAttributes]
                targetSecret = target.loc[tid, Attack.sensitiveAttribute]
                guess = targetSecret
                pCorrect = 1
                mse = 0

                resultsTargetPrivacy[tid][sa]['Raw'][nr]['AttackerGuess'].append(guess)
                resultsTargetPrivacy[tid][sa]['Raw'][nr]['ProbCorrect'].append(pCorrect)
                resultsTargetPrivacy[tid][sa]['Raw'][nr]['MSE'].append(mse)
                resultsTargetPrivacy[tid][sa]['Raw'][nr]['TargetPresence'].append(LABEL_IN)

        ##### Assess advantage Syn
        for GenModel in gmList:
            LOGGER.info(f'Start: Evaluation for model {GenModel.__name__}...')
            GenModel.fit(rawTout)
            synTwithoutTarget = [GenModel.generate_samples(runconfig['sizeSynT']) for _ in range(runconfig['nSynT'])]
            for sa, Attack in attacks.items():

                for tid in targetIDs:
                    resultsTargetPrivacy[tid][sa][GenModel.__name__][nr] = {
                        'AttackerGuess': [],
                        'ProbCorrect': [],
                        'MSE': [],
                        'TargetPresence': [LABEL_OUT for _ in range(runconfig['nSynT'])]
                    }
                for syn in synTwithoutTarget:
                    standSyn = standardize_before_AIA(syn, metadata, scaler)
                    assert not standSyn.isna().any().any()        
                    # scaler.fit(syn.drop('Y', axis = 1))
                    # standSyn = syn
                    # standSyn.iloc[:,:D] = DataFrame(scaler.transform(syn.drop('Y', axis = 1)))
                    Attack.train(standSyn)
                    
                    for tid in targetIDs:
                        target = standTargets.loc[[tid]]
                        targetAux = target.loc[[tid], Attack.knownAttributes]
                        targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                        guess = Attack.attack(targetAux)
                        # don't need to average over nSynT here because this is done in load_results_inference
                        pCorrect = int( targetSecret - runconfig["alpha"] < guess < targetSecret + runconfig["alpha"] )
                        mse = (guess - targetSecret)**2
                        
                        resultsTargetPrivacy[tid][sa][GenModel.__name__][nr]['AttackerGuess'].append(guess)
                        resultsTargetPrivacy[tid][sa][GenModel.__name__][nr]['ProbCorrect'].append(pCorrect)
                        resultsTargetPrivacy[tid][sa][GenModel.__name__][nr]['MSE'].append(mse)

            del synTwithoutTarget
            
            for tid in targetIDs:
                LOGGER.info(f'Target: {tid}')
                target = targets.loc[[tid]]
                rawTin = rawTout.append(target)
                rawTin['Y'] = rawTin['Y'].astype(str)
                GenModel.fit(rawTin)
                synTwithTarget = [GenModel.generate_samples(runconfig['sizeSynT']) for _ in range(runconfig['nSynT'])]

                # for regression
                target = standTargets.loc[[tid]]

                for sa, Attack in attacks.items():
                    targetAux = target.loc[[tid], Attack.knownAttributes]
                    targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                    for syn in synTwithTarget:
                        standSyn = standardize_before_AIA(syn, metadata, scaler)
                        assert  not standSyn.isna().any().any()
                        # scaler.fit(syn.drop('Y', axis = 1))
                        # standSyn = syn
                        # standSyn.iloc[:,:D] = DataFrame(scaler.transform(syn.drop('Y', axis = 1)))
                        Attack.train(standSyn)

                        # for checking coefficients
                        if args.coeff == 1: 
                            new_row = append(Attack.coefficients, [sa, tid, GenModel])
                            regCoeff.append(new_row)

                        guess = Attack.attack(targetAux)
                        
                        # don't need to average over nSynT here because this is done in load_results_inference
                        pCorrect = int( targetSecret - runconfig["alpha"] < guess < targetSecret + runconfig["alpha"] )
                        mse = (guess - targetSecret)**2

                        resultsTargetPrivacy[tid][sa][GenModel.__name__][nr]['AttackerGuess'].append(guess)
                        resultsTargetPrivacy[tid][sa][GenModel.__name__][nr]['ProbCorrect'].append(pCorrect)
                        resultsTargetPrivacy[tid][sa][GenModel.__name__][nr]['MSE'].append(mse)
                        resultsTargetPrivacy[tid][sa][GenModel.__name__][nr]['TargetPresence'].append(LABEL_IN)
            del synTwithTarget


        ##### Assess advantage San
        for San in sanList:
            LOGGER.info(f'Start: Evaluation for sanitiser {San.__name__}...')
            attacks = {}
            for sa, atype in runconfig['sensitiveAttributes'].items():
                if atype == 'LinReg':
                    attacks[sa] = LinRegAttack(sensitiveAttribute=sa, metadata=metadata, quids=San.quids)
                elif atype == 'Classification':
                    attacks[sa] = RandForestAttack(sensitiveAttribute=sa, metadata=metadata, quids=San.quids)

            sanOut = San.sanitise(rawTout)

            for sa, Attack in attacks.items():
                Attack.train(sanOut)

                for tid in targetIDs:
                    target = targets.loc[[tid]]
                    targetAux = target.loc[[tid], Attack.knownAttributes]
                    targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                    guess = Attack.attack(targetAux, attemptLinkage=True, data=sanOut)
                    pCorrect = Attack.get_likelihood(targetAux, targetSecret, runconfig['alpha'], attemptLinkage=True, data=sanOut)

                    resultsTargetPrivacy[tid][sa][San.__name__][nr] = {
                        'AttackerGuess': [guess],
                        'ProbCorrect': [pCorrect],
                        'TargetPresence': [LABEL_OUT]
                }

            for tid in targetIDs:
                LOGGER.info(f'Target: {tid}')
                target = targets.loc[[tid]]
                rawTin = rawTout.append(target)
                sanIn = San.sanitise(rawTin)

                for sa, Attack in attacks.items():
                    targetAux = target.loc[[tid], Attack.knownAttributes]
                    targetSecret = target.loc[tid, Attack.sensitiveAttribute]


                    Attack.train(sanIn)

                    guess = Attack.attack(targetAux, attemptLinkage=True, data=sanIn)
                    pCorrect = Attack.get_likelihood(targetAux, targetSecret, runconfig['alpha'], attemptLinkage=True, data=sanIn)

                    resultsTargetPrivacy[tid][sa][San.__name__][nr]['AttackerGuess'].append(guess)
                    resultsTargetPrivacy[tid][sa][San.__name__][nr]['ProbCorrect'].append(pCorrect)
                    resultsTargetPrivacy[tid][sa][San.__name__][nr]['TargetPresence'].append(LABEL_IN)

    outfile = f"ResultsMLEAI_{dname}"
    LOGGER.info(f"Write results to {path.join(f'{args.outdir}', f'{outfile}')}")

    # for saving regression coefficients
    regCoeff = DataFrame(regCoeff, columns = column_names)
    if args.coeff == 1: 
        regCoeff.to_csv(f'{args.outdir}/reg_coeff_{attacks.keys()}.csv', index = False)

    with open(path.join(f'{args.outdir}', f'{outfile}.json'), 'w') as f:
        json.dump(resultsTargetPrivacy, f, indent=2, default=json_numpy_serialzer)

if __name__ == "__main__":
    main()