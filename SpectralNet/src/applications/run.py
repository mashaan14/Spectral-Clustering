'''
Expected run times on a GTX 1080 GPU:
MNIST: 1 hr
Reuters: 2.5 hrs
cc: 15 min
'''

import sys, os
# add directories in src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import argparse
from collections import defaultdict

from core.data import get_data
from spectralnet import run_net

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu number to use', default='')
# cc circle238 Sparse622
parser.add_argument('--dset', type=str, help='gpu number to use', default='Comp399')
args = parser.parse_args()

# SELECT GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

params = defaultdict(lambda: None)

# SET GENERAL HYPERPARAMETERS
general_params = {
        'dset': args.dset,                  # dataset: reuters / mnist
        'val_set_fraction': 0.1,            # fraction of training set to use as validation
        'precomputedKNNPath': '',           # path for precomputed nearest neighbors (with indices and saved as a pickle or h5py file)
        'siam_batch_size': 128,             # minibatch size for siamese net
        }
params.update(general_params)

# SET DATASET SPECIFIC HYPERPARAMETERS
# for the original run.py file please refer to original code at https://github.com/KlugerLab/SpectralNet

# HYPERPARAMETERS EXPLAINED
# if args.dset == 'mnist':
#     mnist_params = {
#         'n_clusters': 10,                   # number of clusters in data
#         'use_code_space': True,             # enable / disable code space embedding
#         'affinity': 'siamese',              # affinity type: siamese / knn
#         'n_nbrs': 3,                        # number of nonzero entries (neighbors) to use for graph Laplacian affinity matrix
#         'scale_nbr': 2,                     # neighbor used to determine scale of gaussian graph Laplacian; calculated by
#                                             # taking median distance of the (scale_nbr)th neighbor, over a set of size batch_size
#                                             # sampled from the datset

#         'siam_k': 2,                        # threshold where, for all k <= siam_k closest neighbors to x_i, (x_i, k) is considered
#                                             # a 'positive' pair by siamese net

#         'siam_ne': 400,                     # number of training epochs for siamese net
#         'spec_ne': 400,                     # number of training epochs for spectral net
#         'siam_lr': 1e-3,                    # initial learning rate for siamese net
#         'spec_lr': 1e-3,                    # initial learning rate for spectral net
#         'siam_patience': 10,                # early stopping patience for siamese net
#         'spec_patience': 20,                # early stopping patience for spectral net
#         'siam_drop': 0.1,                   # learning rate scheduler decay for siamese net
#         'spec_drop': 0.1,                   # learning rate scheduler decay for spectral net
#         'batch_size': 1024,                 # batch size for spectral net
#         'siam_reg': None,                   # regularization parameter for siamese net
#         'spec_reg': None,                   # regularization parameter for spectral net
#         'siam_n': None,                     # subset of the dataset used to construct training pairs for siamese net
#         'siamese_tot_pairs': 600000,        # total number of pairs for siamese net
#         'arch': [                           # network architecture. if different architectures are desired for siamese net and
#                                             #   spectral net, 'siam_arch' and 'spec_arch' keys can be used
#             {'type': 'relu', 'size': 1024},
#             {'type': 'relu', 'size': 1024},
#             {'type': 'relu', 'size': 512},
#             {'type': 'relu', 'size': 10},
#             ],
#         'use_approx': False,                # enable / disable approximate nearest neighbors
#         'use_all_data': True,               # enable to use all data for training (no test set)
#         }
#     params.update(mnist_params)

if args.dset == 'cc':
    cc_params = {
        # data generation parameters
        'train_set_fraction': 1.,       # fraction of the dataset to use for training
        'noise_sig': 0.1,               # variance of the gaussian noise applied to x
        'n': 1500,                      # number of total points in dataset
        # training parameters
        'n_clusters': 2,
        'use_code_space': False,
        # 'affinity': 'full',
        ########################
        # speceific parameters for mu0 graph        
        'affinity': 'siamese',
        'n_nbrs': 5,
        'scale_nbr': 4,
        'siam_k': 2,
        'siam_ne': 100,
        'siam_lr': 1e-3,
        'siam_patience': 10,
        'siam_drop': 0.1,
        'siam_reg': None,
        'siam_n': None,
        'use_mu0': True,
        'use_approx': False,
        ########################
        'spec_ne': 300,
        'spec_lr': 1e-3,
        'spec_patience': 30,
        'spec_drop': 0.1,
        'batch_size': 128,
        'batch_size_orthonorm': 128,
        'spec_reg': None,
        'arch': [
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            ],
        'use_all_data': True,
        }
    params.update(cc_params)
elif args.dset == 'Comp399':
    circle238_params = {
        # data generation parameters
        'train_set_fraction': 1.,       # fraction of the dataset to use for training
        'noise_sig': 0,               # variance of the gaussian noise applied to x
        'n': 399,                      # number of total points in dataset
        # training parameters
        'n_clusters': 6,
        'use_code_space': False,
        # 'affinity': 'full',
        ########################
        # speceific parameters for mu0 graph        
        'affinity': 'siamese',
        'n_nbrs': 5,
        'scale_nbr': 4,
        'siam_k': 2,
        'siam_ne': 1,
        'siam_lr': 1e-3,
        'siam_patience': 10,
        'siam_drop': 0.1,
        'siam_reg': None,
        'siam_n': None,
        'use_mu0': True,
        'use_approx': False,
        ########################
        'spec_ne': 1,
        'spec_lr': 1e-3,
        'spec_patience': 300,
        'spec_drop': 0.1,
        'batch_size': 128,
        'batch_size_orthonorm': 128,
        'spec_reg': None,
        'arch': [
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            ],
        'generalization_metrics': False,
        'use_all_data': True,
        }
    params.update(circle238_params)
elif args.dset == 'Agg788':
    Agg788_params = {
        # data generation parameters
        'train_set_fraction': 1.,       # fraction of the dataset to use for training
        'noise_sig': 0,               # variance of the gaussian noise applied to x
        'n': 788,                      # number of total points in dataset
        # training parameters
        'n_clusters': 7,
        'use_code_space': False,
        # 'affinity': 'full',
        ########################
        # speceific parameters for mu0 graph        
        'affinity': 'siamese',
        'n_nbrs': 5,
        'scale_nbr': 4,
        'siam_k': 2,
        'siam_ne': 100,
        'siam_lr': 1e-3,
        'siam_patience': 10,
        'siam_drop': 0.1,
        'siam_reg': None,
        'siam_n': None,
        'use_mu0': True,
        'use_approx': False,
        ########################
        'spec_ne': 300,
        'spec_lr': 1e-3,
        'spec_patience': 300,
        'spec_drop': 0.1,
        'batch_size': 128,
        'batch_size_orthonorm': 128,
        'spec_reg': None,
        'arch': [
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            ],
        'generalization_metrics': False,
        'use_all_data': True,
        }
    params.update(Agg788_params)        

# LOAD DATA
data = get_data(params)

# RUN EXPERIMENT
x_spectralnet, y_spectralnet = run_net(data, params)

if args.dset in ['cc', 'circle238', 'Sparse622', 'Agg788', 'Comp399']:
    # run plotting script
    import plot_2d
    plot_2d.process(x_spectralnet, y_spectralnet, data, params)
