# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import copy
import os
import sys

import dnnlib
from dnnlib import EasyDict

from metrics.metric_defaults import metric_defaults

#----------------------------------------------------------------------------

_valid_configs = [
    # Table 1
    'config-a', # Baseline StyleGAN
    'config-b', # + Weight demodulation
    'config-c', # + Lazy regularization
    'config-d', # + Path length regularization
    'config-e', # + No growing, new G & D arch.
    'config-f', # + Large networks (default)

    # Table 2
    'config-e-Gorig-Dorig',   'config-e-Gorig-Dresnet',   'config-e-Gorig-Dskip',
    'config-e-Gresnet-Dorig', 'config-e-Gresnet-Dresnet', 'config-e-Gresnet-Dskip',
    'config-e-Gskip-Dorig',   'config-e-Gskip-Dresnet',   'config-e-Gskip-Dskip',
]

#----------------------------------------------------------------------------

def run(dataset, data_dir, result_dir, config_id, num_gpus, gamma, mirror_augment, metrics, resume_pkl,
    minibatch_gpu,
    data_size,
    num_epochs,
    init_proj_dim,
    init_staleness,
    num_samples_factor,
    knn_perturb_factor,
    candidate_batch_size,
    exclusive_retrieved_code,
    NN_rec_lpips_weight,
    dist_thres_percentile,
    attr_interesting,
    init_mul,
    ):
    train     = EasyDict(run_func_name='training.training_loop.training_loop') # Options for training loop.
    G         = EasyDict(func_name='training.networks_stylegan2.G_main', init_mul=init_mul)       # Options for generator network.
    D         = EasyDict(func_name='training.networks_stylegan2.D_stylegan2_feature')  # Options for discriminator network.
    G_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for generator optimizer.
    D_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for discriminator optimizer.
    G_loss    = EasyDict(func_name='training.loss.G_logistic_ns_rec_interp_arb_pathreg', NN_rec_lpips_weight=NN_rec_lpips_weight)      # Options for generator loss.
    D_loss    = EasyDict(func_name='training.loss.D_logistic_r1')              # Options for discriminator loss.
    sched     = EasyDict()                                                     # Options for TrainingSchedule.
    grid      = EasyDict(size='1080p', layout='random')                        # Options for setup_snapshot_image_grid().
    sc        = dnnlib.SubmitConfig()                                          # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': 1000, 'gpu_options.allow_growth': False, 'graph_options.place_pruned_graph': True} # Options for tflib.init_tf().

    train.data_dir = data_dir
    train.total_kimg = (data_size * num_epochs) // 1000
    train.mirror_augment = mirror_augment
    sched.G_lrate_base = sched.D_lrate_base = 0.002
    sched.minibatch_gpu_base = minibatch_gpu
    sched.minibatch_size_base = sched.minibatch_gpu_base * num_gpus
    D_loss.gamma = 10.0
    metrics = [metric_defaults[x] for x in metrics]
    train.resume_pkl = resume_pkl

    train.data_size = data_size
    train.num_epochs = num_epochs
    train.init_proj_dim = init_proj_dim
    train.init_staleness = init_staleness
    train.num_samples_factor = num_samples_factor
    train.knn_perturb_factor = knn_perturb_factor
    train.candidate_batch_size = candidate_batch_size
    train.exclusive_retrieved_code = exclusive_retrieved_code
    train.dist_thres_percentile = dist_thres_percentile
    train.attr_interesting = attr_interesting

    desc = 'stylegan2'

    desc += '-' + dataset
    dataset_args = EasyDict(tfrecord_dir=dataset, max_label_size='full')

    #assert num_gpus in [1, 2, 4, 8]
    sc.num_gpus = num_gpus
    desc += '-%dgpu' % num_gpus

    assert config_id in _valid_configs
    desc += '-' + config_id

    if init_proj_dim is None:
        desc += '_noProj'
    else:
        desc += '_%dProj' % init_proj_dim
    desc += '_init_staleness_%d' % init_staleness
    desc += '_num_samples_factor_%d' % num_samples_factor
    desc += '_knn_perturb_factor_%f' % knn_perturb_factor
    #if exclusive_retrieved_code:
    #    desc += '_exclusive_retrieved_code'
    #else:
    #    desc += '_NONexclusive_retrieved_code'
    desc += '_NN_rec_lpips_weight_%f' % NN_rec_lpips_weight
    if attr_interesting is not None:
        desc += '_%s' % attr_interesting.replace(',', '_and_')
    if resume_pkl is None or '_scratch' in resume_pkl:
        desc += '_scratch'
    else:
        desc += '_finetune'

    # Configs A-E: Shrink networks to match original StyleGAN.
    if config_id != 'config-f':
        G.fmap_base = D.fmap_base = 8 << 10

    # Config E: Set gamma to 100 and override G & D architecture.
    if config_id.startswith('config-e'):
        D_loss.gamma = 100
        if 'Gorig'   in config_id: G.architecture = 'orig'
        if 'Gskip'   in config_id: G.architecture = 'skip' # (default)
        if 'Gresnet' in config_id: G.architecture = 'resnet'
        if 'Dorig'   in config_id: D.architecture = 'orig'
        if 'Dskip'   in config_id: D.architecture = 'skip'
        if 'Dresnet' in config_id: D.architecture = 'resnet' # (default)

    # Configs A-D: Enable progressive growing and switch to networks that support it.
    if config_id in ['config-a', 'config-b', 'config-c', 'config-d']:
        sched.lod_initial_resolution = 8
        sched.G_lrate_base = sched.D_lrate_base = 0.001
        sched.G_lrate_dict = sched.D_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        sched.minibatch_size_base = 32 # (default)
        sched.minibatch_size_dict = {8: 256, 16: 128, 32: 64, 64: 32}
        sched.minibatch_gpu_base = 4 # (default)
        sched.minibatch_gpu_dict = {8: 32, 16: 16, 32: 8, 64: 4}
        G.synthesis_func = 'G_synthesis_stylegan_revised'
        D.func_name = 'training.networks_stylegan2.D_stylegan'

    # Configs A-C: Disable path length regularization.
    if config_id in ['config-a', 'config-b', 'config-c']:
        G_loss = EasyDict(func_name='training.loss.G_logistic_ns')

    # Configs A-B: Disable lazy regularization.
    if config_id in ['config-a', 'config-b']:
        train.lazy_regularization = False

    # Config A: Switch to original StyleGAN networks.
    if config_id == 'config-a':
        G = EasyDict(func_name='training.networks_stylegan.G_style')
        D = EasyDict(func_name='training.networks_stylegan.D_basic')

    if gamma is not None:
        D_loss.gamma = gamma

    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    kwargs = EasyDict(train)
    kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, G_loss_args=G_loss, D_loss_args=D_loss)
    kwargs.update(dataset_args=dataset_args, sched_args=sched, grid_args=grid, metric_arg_list=metrics, tf_config=tf_config)
    kwargs.submit_config = copy.deepcopy(sc)
    kwargs.submit_config.run_dir_root = result_dir
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)

#----------------------------------------------------------------------------

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _parse_comma_sep(s):
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train StyleGAN2.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    parser.add_argument('--dataset', help='Training dataset', required=True)
    parser.add_argument('--config', help='Training config (default: %(default)s)', default='config-e', dest='config_id', metavar='CONFIG')
    parser.add_argument('--init-mul', help='std multiplier for weight initialization (default: %(default)s)', default=1.0, type=float)
    parser.add_argument('--num-gpus', help='Number of GPUs (default: %(default)s)', default=1, type=int, metavar='N')
    parser.add_argument('--gamma', help='R1 regularization weight (default is config dependent)', default=None, type=float)
    parser.add_argument('--mirror-augment', help='Mirror augment (default: %(default)s)', default=False, metavar='BOOL', type=_str_to_bool)
    parser.add_argument('--metrics', help='Comma-separated list of metrics or "none" (default: %(default)s)', default='fid50k', type=_parse_comma_sep)
    parser.add_argument('--minibatch-gpu', help='Batch size per GPU (default: %(default)s)', metavar='N', default=6, type=int)

    parser.add_argument('--data-size', help='Number of images in the dataset (default: %(default)s)', metavar='N', default=30000, type=int)
    parser.add_argument('--num-epochs', help='Number of epochs in total (default: %(default)s)', metavar='N', default=10000, type=int)

    parser.add_argument('--init-proj-dim', help='Dimension of random projection for DCI NN search (default is without projection)', metavar='N', default=None, type=int)
    parser.add_argument('--init-staleness', help='Initial epoch multiplier to update NN candidates in IMLE (grows exponentially in the power of 2) (default: %(default)s)', metavar='N', default=10, type=int)
    parser.add_argument('--num-samples-factor', help='The multiplier of number of NN candidates in IMLE w.r.t. data size (default: %(default)s)', metavar='N', default=10, type=int)
    parser.add_argument('--knn-perturb-factor', help='Std of a normal perturbation added to the 1NN result in IMLE (default: %(default)s)', default=0.05, type=float)
    parser.add_argument('--candidate-batch-size', help='Batch size for NN candidate feature calculation in IMLE (default: %(default)s)', metavar='N', default=256, type=int)
    parser.add_argument('--exclusive-retrieved-code', help='Enforce uniqueness of IMLE retrieved NN latent code? (default is not to enforce)', metavar='N', default=0, type=int)

    parser.add_argument('--NN-rec-lpips-weight', help='Weight for LPIPS-distance-based reconstruction loss of 1NN pairs(default: %(default)s)', default=2.5, type=float)
    parser.add_argument('--dist-thres-percentile', help='Percentile of NN pairs, based on latent reconstruction distances, to be considered for IMLE (default: %(default)s)', default=100.0, type=float)
    parser.add_argument('--attr-interesting', help='The interesting CelebA attributes of a minority subgroup (default is to consider the entire dataset)', default=None, type=str)

    parser.add_argument('--resume-pkl', help='Pretrained network pkl path (default is to train from scratch)', default=None, type=str)

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print ('Error: dataset root directory does not exist.')
        sys.exit(1)

    if args.config_id not in _valid_configs:
        print ('Error: --config value must be one of: ', ', '.join(_valid_configs))
        sys.exit(1)

    for metric in args.metrics:
        if metric not in metric_defaults:
            print ('Error: unknown metric \'%s\'' % metric)
            sys.exit(1)

    run(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------

