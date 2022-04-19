"""
Advanced Machine Learning, 2022
HW 4 Base Code

Author: Andrew H. Fagg (andrewhfagg@gmail.com)

Image classification

"""

import argparse
import pickle
import pandas as pd
import numpy as np
import os
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from tensorflow.python.keras.utils.vis_utils import model_to_dot
from tensorflow.python.keras.utils.vis_utils import plot_model

from job_control import *
from pfam_loader import *
from create_network import *

#################################################################
# Default plotting parameters
FIGURESIZE = (10, 6)
FONTSIZE = 18

plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE


#################################################################
def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='CNN', fromfile_prefix_chars='@')

    # High-level commands
    parser.add_argument('--check', action='store_true', help='Check results for completeness')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")

    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')

    # High-level experiment configuration
    parser.add_argument('--exp_type', type=str, default=None, help="Experiment type")
    parser.add_argument('--label', type=str, default=None, help="Extra label to add to output files")
    parser.add_argument('--dataset', type=str, default='/home/fagg/datasets/pfam',
                        help='Data set directory')
    parser.add_argument('--allele', type=str, default='1301', help="Allele number to focus on")
    parser.add_argument('--Nfolds', type=int, default=5, help='Maximum number of folds')
    parser.add_argument('--results_path', type=str, default='./results', help='Results directory')

    # Specific experiment configuration
    parser.add_argument('--exp_index', type=int, default=1, help='Experiment index')
    parser.add_argument('--rotation', type=int, default=0, help='Cross-validation rotation')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--Ntraining', type=int, default=3, help='Number of training folds')
    parser.add_argument('--lrate', type=float, default=0.001, help="Learning rate")

    # Optimization parameters
    parser.add_argument('--decay', type=float, default=0.0, help="Adam decay rate")

    # RNN parameters
    parser.add_argument('--embedding_length', type=int, default=4,
                        help='Length of embedding vector')
    parser.add_argument('--rnnNeurons', type=int, default=10, help='Number of neurons in RNN module')
    parser.add_argument('--rnn_activation', type=str, default=None, help='Activation of RNN layers')
    parser.add_argument('--r_drop', type=float, default=0.0, help='Recurrent Dropout')

    # CNN parameters
    parser.add_argument('--filters', nargs='+', type=int, default=[64],
                        help='Number of filters per 1D-CNN')

    # Hidden unit parameters
    parser.add_argument('--hidden', nargs='+', type=int, default=[100, 5],
                        help='Number of hidden units per layer (sequence of ints)')
    parser.add_argument('--hidden_activation', type=str, default='elu',
                        help='Activation of hidden layers')

    # Regularization parameters
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('--spatial_dropout', type=float, default=None, help='Conv dropout')
    parser.add_argument('--L1_regularizer', '--l1', type=float, default=None, help="L1 regularization parameter")
    parser.add_argument('--L2_regularizer', '--l2', type=float, default=None, help="L2 regularization parameter")

    # Early stopping
    parser.add_argument('--min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early termination")

    # Training parameters
    parser.add_argument('--batch', type=int, default=64, help="Training set batch size")
    parser.add_argument('--steps_per_epoch', type=int, default=10, help="Number of gradient descent steps per epoch")
    parser.add_argument('--validation_fraction', type=float, default=0.25,
                        help="Fraction of available validation set to actually use for validation")
    parser.add_argument('--testing_fraction', type=float, default=0.5,
                        help="Fraction of available testing set to actually use for testing")
    parser.add_argument('--generator_seed', type=int, default=42, help="Seed used for generator configuration")


    return parser


def exp_type_to_hyperparameters(args):
    '''
    Translate the exp_type into a hyperparameter set

    :param args: ArgumentParser
    :return: Hyperparameter set (in dictionary form)
    '''
    if args.exp_type is None:
        p = {'rotation': range(5)}
    else:
        assert False, "Unrecognized exp_type"

    return p


#################################################################
def check_args(args):
    assert (args.rotation >= 0 and args.rotation < args.Nfolds), "Rotation must be between 0 and Nfolds"
    assert (args.Ntraining >= 1 and args.Ntraining <= (args.Nfolds - 1)), "Ntraining must be between 1 and Nfolds-2"
    assert (args.dropout is None or (args.dropout > 0.0 and args.dropout < 1)), "Dropout must be between 0 and 1"
    assert (args.lrate > 0.0 and args.lrate < 1), "Lrate must be between 0 and 1"
    assert (args.L1_regularizer is None or (
            args.L1_regularizer > 0.0 and args.L1_regularizer < 1)), "L2_regularizer must be between 0 and 1"
    assert (args.L2_regularizer is None or (
            args.L2_regularizer > 0.0 and args.L2_regularizer < 1)), "L2_regularizer must be between 0 and 1"
    assert (args.cpus_per_task is None or args.cpus_per_task > 1), "cpus_per_task must be positive or None"


def augment_args(args):
    '''
    Use the jobiterator to override the specified arguments based on the experiment index.

    Modifies the args

    :param args: arguments from ArgumentParser
    :return: A string representing the selection of parameters to be used in the file name
    '''

    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be executing
    p = exp_type_to_hyperparameters(args)

    # Check index number
    index = args.exp_index
    if (index is None):
        return ""

    # Create the iterator
    ji = JobIterator(p)
    print("Total jobs:", ji.get_njobs())

    # Check bounds
    assert (args.exp_index >= 0 and args.exp_index < ji.get_njobs()), "exp_index out of range"

    # Print the parameters specific to this exp_index
    print(ji.get_index(args.exp_index))

    # Push the attributes to the args object and return a string that describes these structures
    return ji.set_attributes_by_index(args.exp_index, args)


#################################################################

def generate_fname(args, params_str):
    '''
    Generate the base file name for output files/directories.

    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.

    :param args: from argParse
    :params_str: String generated by the JobIterator
    '''
    # Hidden unit configuration
    hidden_str = '_'.join(str(x) for x in args.hidden)

    # Dropout
    if args.dropout is None:
        dropout_str = ''
    else:
        dropout_str = 'drop_%0.3f_' % (args.dropout)

    # L1 regularization
    if args.L1_regularizer is None:
        regularizer_l1_str = ''
    else:
        regularizer_l1_str = 'L1_%0.6f_' % (args.L1_regularizer)

    # L2 regularization
    if args.L2_regularizer is None:
        regularizer_l2_str = ''
    else:
        regularizer_l2_str = 'L2_%0.6f_' % (args.L2_regularizer)

    # Label
    if args.label is None:
        label_str = ""
    else:
        label_str = "%s_" % args.label

    # Experiment type
    if args.exp_type is None:
        experiment_type_str = ""
    else:
        experiment_type_str = "%s_" % args.exp_type

    # Epochs
    if args.epochs is None:
        epochs_str = ""
    else:
        epochs_str = "%d_" % args.epochs

    # learning rate
    lrate_str = "LR_%0.6f_" % args.lrate

    fname = "%s/amino_%s%s_epochs_%s_hidden_%s_%s%s%s%sntrain_%02d_rot_%02d" % (
        args.results_path,
        experiment_type_str,
        label_str,
        epochs_str,
        hidden_str,
        dropout_str,
        regularizer_l1_str,
        regularizer_l2_str,
        lrate_str,
        args.Ntraining,
        args.rotation)

    # Replace all but the first dot with a dash
    fnameNew = fname[0] + fname[1:].replace(".", "-")
    #fname = fname.replace(".", "DOT")

    # Put it all together, including #of training folds and the experiment rotation
    return fnameNew

#################################################################
def execute_exp(args=None):
    '''
    Perform the training and evaluation for a single model

    :param args: Argparse arguments
    '''

    # Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])

    print(args.exp_index)

    # Override arguments if we are using exp_index

    args_str = augment_args(args)
    print('Passed augment_args')

    # Set number of threads, if it is specified
    if args.cpus_per_task is not None:
        # Makes sure that you are using no more than the number of threads that you asked for when set up batch file
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)
    print('Passed configure cpus')

    dat_out = load_rotation(basedir=args.dataset, rotation=args.exp_index)
    #dat_out = prepare_data_set(basedir=args.dataset, rotation=args.exp_index)

    # Compute the number of samples in each data set
    nsamples_train = dat_out['ins_train'].size
    nsamples_validation = dat_out['ins_valid'].size
    if dat_out['ins_test'] is None:
        nsamples_testing = 0
    else:
        nsamples_testing = dat_out['ins_test'].size

    print("Total samples: Tr:%d, V:%d, Te:%d" % (nsamples_train, nsamples_validation, nsamples_testing))

    # Essentially, each layer is a dictionary with a given set of properties.
    dense_layers = [{'units': i} for i in args.hidden]

    print("Dense layers:", dense_layers)



    model = create_network(outs=dat_out['outs_train'],
                           vocab_size=dat_out['n_tokens'],
                           output_dim=args.embedding_length,
                           len_max=dat_out['len_max'],
                           dense_layers=dense_layers,
                           n_neurons=args.rnnNeurons,
                           activation=args.rnn_activation,
                           activation_dense=args.hidden_activation,
                           lambda_regularization=None,
                           use_gru=False,
                           dropout=args.dropout,
                           r_drop=args.r_drop,
                           lrate=args.lrate)

    # Report model structure if verbosity is turned on
    if args.verbose >= 1:
        print(model.summary())

    print(args)

    # Output file base and pkl file
    fbase = generate_fname(args, args_str)
    fname_out = "%s_results.pkl" % fbase

    # Perform the experiment?
    if (args.nogo):
        # No!
        print("NO GO")
        print(fbase)
        return

    # Check if output file already exists
    if os.path.exists(fname_out):
        # Results file does exist: exit
        print("File %s already exists" % fname_out)
        return

    # Callbacks
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                                                         mode='min',
                                                         patience=args.patience,
                                                         restore_best_weights=True,
                                                         min_delta=args.min_delta)

    #dat_train, dat_valid, dat_test = create_tf_datasets(dat_out, batch=args.batch)

    history = model.fit(x=dat_out['ins_train'],
                        y=dat_out['outs_train'],
                        batch_size=args.batch,
                        epochs=args.epochs,
                        use_multiprocessing=False,
                        verbose=args.verbose >= 2,
                        validation_data=(dat_out['ins_valid'], dat_out['outs_valid']),
                        validation_steps=None,
                        callbacks=[early_stopping_cb])

    print(model.summary())


    # Generate results data
    results = {}
    results['args'] = args
    results['predict_validation'] = model.predict(dat_out['ins_valid'])
    results['predict_validation_eval'] = model.evaluate(dat_out['ins_valid'], dat_out['outs_valid'])

    if dat_out['ins_test'] is not None:
        results['predict_testing'] = model.predict(dat_out['ins_test'])
        results['predict_testing_eval'] = model.evaluate(dat_out['ins_test'], dat_out['outs_test'])

    results['predict_training'] = model.predict(dat_out['ins_train'])
    results['predict_training_eval'] = model.evaluate(dat_out['ins_train'], dat_out['outs_train'])
    results['history'] = history.history
    tf.keras.utils.plot_model(model, to_file='%s_model_plot.png' % fbase, show_shapes=True, show_layer_names=True)


    # Save results
    fbase = generate_fname(args, args_str)
    results['fname_base'] = fbase
    with open("%s_results.pkl" % (fbase), "wb") as fp:
        pickle.dump(results, fp)

    # Save model
    model.save("%s_model" % (fbase))

    print(fbase)

    return model


def check_completeness(args):
    '''
    Check the completeness of a Cartesian product run.

    All other args should be the same as if you executed your batch, however, the '--check' flag has been set

    Prints a report of the missing runs, including both the exp_index and the name of the missing results file

    :param args: ArgumentParser

    '''

    # Get the corresponding hyperparameters
    p = exp_type_to_hyperparameters(args)

    # Create the iterator
    ji = JobIterator(p)

    print("Total jobs: %d" % ji.get_njobs())

    print("MISSING RUNS:")

    indices = []
    # Iterate over all possible jobs
    for i in range(ji.get_njobs()):
        params_str = ji.set_attributes_by_index(i, args)
        # Compute output file name base
        fbase = generate_fname(args, params_str)

        # Output pickle file name
        fname_out = "%s_results.pkl" % (fbase)

        if not os.path.exists(fname_out):
            # Results file does not exist: report it
            print("%3d\t%s" % (i, fname_out))
            indices.append(i)

    # Give the list of indices that can be inserted into the --array line of the batch file
    print("Missing indices (%d): %s" % (len(indices), ','.join(str(x) for x in indices)))


#################################################################
if __name__ == "__main__":
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    # Turn off GPU?
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # GPU check
    physical_devices = tf.config.list_physical_devices('GPU')
    n_physical_devices = len(physical_devices)
    if (n_physical_devices > 0):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print('We have %d GPUs\n' % n_physical_devices)
    else:
        print('NO GPU')

    if (args.check):
        # Just check to see if all experiments have been executed
        check_completeness(args)
    else:
        # Execute the experiment
        execute_exp(args)
