'''
PFAM Data loader and data set constructor

Author: Andrew H. Fagg


Two different ways to load full data sets:

prepare_data_set(basedir = '/home/fagg/datasets/pfam', rotation = 0, nfolds = 5, ntrain_folds = 3)
    loads the raw CSV files, does the splitting and tokenization

OR

load_rotation(basedir = '/home/fagg/datasets/pfam', rotation=0)
    loads an already stored data set from a pickle file




'''
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
import random
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow import keras

def load_pfam_file(basedir, fold):
    '''
    Load a CSV file into a DataFrame
    :param basedir: Directory containing input files
    :param fold: Fold to load
    '''
    
    df = pd.read_csv('%s/pfam_fold_%d.csv'%(basedir, fold))
    return df

def load_pfam_dataset(basedir = '/home/fagg/datasets/pfam', rotation = 0, nfolds = 5, ntrain_folds = 3):
    '''
    Load train/valid/test datasets into DataFrames

    :param basedir: Directory containing input files
    :param rotation: Rotation to load
    :param nfolds: Total number of folds
    :param ntrain_folds: Number of training folds to use

    :return: Dictionary containing the DataFrames
    '''

    train_folds = (np.arange(ntrain_folds) + rotation)  % nfolds
    valid_folds = (np.array([ntrain_folds]) + rotation) % nfolds
    test_folds = (np.array([ntrain_folds]) + 1 + rotation) % nfolds

    train_dfs = [load_pfam_file(basedir, f) for f in train_folds]
    valid_dfs = [load_pfam_file(basedir, f) for f in valid_folds]
    test_dfs = [load_pfam_file(basedir, f) for f in test_folds]

    train_df = pd.concat(train_dfs, ignore_index=True)
    valid_df = pd.concat(valid_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    return {'train': train_df, 'valid': valid_df, 'test': test_df}


def prepare_data_set(basedir = '/home/fagg/datasets/pfam', rotation = 0, nfolds = 5, ntrain_folds = 3):
    '''
    Generate a full data set

    :param basedir: Directory containing input files
    :param rotation: Rotation to load
    :param nfolds: Total number of folds
    :param ntrain_folds: Number of training folds to use

    :return: Dictionary containing a full train/validation/test data set

    Dictionary format:
    ins_train: tokenized training inputs (examples x len_max)
    outs_train: tokenized training outputs (examples x 1).  Values are 0 ... n_tokens-1
    ins_valid: tokenized validation inputs (examples x len_max)
    outs_valid: tokenized validation outputs (examples x 1)
    ins_test: tokenized test inputs (examples x len_max)
    outs_test: tokenized test outputs (examples x 1)
    len_max: maximum length of a string
    n_tokens: Maximum number of output tokens 
    out_index_word: dictionary containing index -> class name map (note index is 1... n_toeksn)
    out_word_index: dictionary containing class name -> index map (note index is 1... n_toeksn)
    '''

    
    # Load the data from the disk
    dat = load_pfam_dataset(basedir=basedir, rotation=rotation, nfolds=nfolds, ntrain_folds=ntrain_folds)

    # Extract ins/outs
    dat_out = {}

    # Extract ins/outs for each dataset
    for k, df in dat.items():
        # Get the set of strings
        
        dat_out['ins_'+k] = df['string'].values
        dat_out['outs_'+k] = df['label'].values

    # Compute max length: only defined with respect to the training set
    len_max = np.max(np.array([len(s) for s in dat_out['ins_train']]))

    # TODO: Remove once testing complete
    test = pd.DataFrame(dat_out['outs_test'])

    print('tokenize fit...')
    # Convert strings to lists of indices
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True,
                                                   filters='\t\n')
    tokenizer.fit_on_texts(dat_out['ins_train'])

    print('tokenize...')
    # Loop over all data sets
    for k in dat.keys():
        # Loop over all strings and tokenize
        seq = tokenizer.texts_to_sequences(dat_out['ins_'+k])
        dat_out['ins_'+k] = pad_sequences(seq, maxlen=len_max) # Pad out so all are the same length

    n_tokens = np.max(dat_out['ins_train']) + 2

    print('outputs...')
    # Loop over all data sets: create tokenizer for output
    tokenizer = keras.preprocessing.text.Tokenizer(filters='\t\n')
    tokenizer.fit_on_texts(dat_out['outs_train']) # Essentially turns into label encoding

    # Tokenize all of the outputs
    for k in dat.keys():
        dat_out['outs_'+k] = np.array(tokenizer.texts_to_sequences(dat_out['outs_'+k]))-1
        #np.expand_dims(dat_out['outs_'+k],  axis=-1)seq =
        
    #
    dat_out['len_max'] = len_max
    dat_out['n_tokens'] = n_tokens
    dat_out['out_index_word'] = tokenizer.index_word
    dat_out['out_word_index'] = tokenizer.word_index
    dat_out['rotation'] = rotation
    
    return dat_out

    
def save_data_sets(basedir = '/home/fagg/datasets/pfam', out_basedir = None, nfolds = 5, ntrain_folds = 3):
    '''
    Generate pickle files for all rotations.

    :param basedir: Directory containing input files
    :param out_basedir: Directory for output files (None -> use the basedir)
    :param nfolds: Total number of folds
    :param ntrain_folds: Number of training folds to use
    :param rotation: Rotation to load

    :return: Dictionary containing a full train/validation/test data set
    '''

    if out_basedir is None:
        out_basedir = basedir
        
    # Loop over all rotations
    for r in range(nfolds):
        # Load the rotation
        dat=prepare_data_set(basedir=basedir, rotation=r, nfolds=nfolds, ntrain_folds=ntrain_folds)

        # Write rotation to pickle file
        fname = '%s/pfam_rotation_%d.pkl'%(basedir, r)

        with open(fname, 'wb') as fp:
            pickle.dump(dat, fp)
            
def load_rotation(basedir = '/home/fagg/datasets/pfam', rotation=0):
    '''
    Load a single rotation from a pickle file.  These rotations are 5 folds, 3 training folds

    :param basedir: Directory containing files
    :param rotation: Rotation to load

    :return: Dictionary containing a full train/validation/test data set
    '''
    fname = '%s/pfam_rotation_%d.pkl'%(basedir, rotation)
    with open(fname, 'rb') as fp:
        dat_out = pickle.load(fp)
        return dat_out
    return None
