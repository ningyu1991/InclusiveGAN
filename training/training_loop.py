# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Main training script."""

import os
import numpy as np
import tensorflow as tf
import sklearn
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

from training import dataset
from training import misc
from metrics import metric_base

import sys
sys.path.append('./dci_code')
from dci import DCI

#----------------------------------------------------------------------------
# Function to determine the dimension of random projection.

def func_proj_dim(init_proj_dim, data_size, num_samples_factor, G):
    if init_proj_dim is None:
        proj_dim = np.prod(G.output_shape[1:])
    elif init_proj_dim == 0:
        proj_dim = sklearn.random_projection.johnson_lindenstrauss_min_dim(n_samples=data_size*num_samples_factor, eps=0.1)
    else:
        proj_dim = init_proj_dim
    return proj_dim

#----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

def process_reals(x, labels, lod, mirror_augment, drange_data, drange_net):
    with tf.name_scope('DynamicRange'):
        x = tf.cast(x, tf.float32)
        x = misc.adjust_dynamic_range(x, drange_data, drange_net)
    if mirror_augment:
        with tf.name_scope('MirrorAugment'):
            x = tf.where(tf.random_uniform([tf.shape(x)[0]]) < 0.5, x, tf.reverse(x, [3]))
    with tf.name_scope('FadeLOD'): # Smooth crossfade between consecutive levels-of-detail.
        s = tf.shape(x)
        y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
        y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
        y = tf.tile(y, [1, 1, 1, 2, 1, 2])
        y = tf.reshape(y, [-1, s[1], s[2], s[3]])
        x = tflib.lerp(x, y, lod - tf.floor(lod))
    with tf.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
        s = tf.shape(x)
        factor = tf.cast(2 ** tf.floor(lod), tf.int32)
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x, labels

#----------------------------------------------------------------------------
# Evaluate time-varying training parameters.

def training_schedule(
    cur_nimg,
    training_set,
    lod_initial_resolution  = None,     # Image resolution used at the beginning.
    lod_training_kimg       = 600,      # Thousands of real images to show before doubling the resolution.
    lod_transition_kimg     = 600,      # Thousands of real images to show when fading in new layers.
    minibatch_size_base     = 64,       # Global minibatch size.
    minibatch_size_dict     = {},       # Resolution-specific overrides.
    minibatch_gpu_base      = 32,       # Number of samples processed at a time by one GPU.
    minibatch_gpu_dict      = {},       # Resolution-specific overrides.
    G_lrate_base            = 0.002,    # Learning rate for the generator.
    G_lrate_dict            = {},       # Resolution-specific overrides.
    D_lrate_base            = 0.002,    # Learning rate for the discriminator.
    D_lrate_dict            = {},       # Resolution-specific overrides.
    lrate_rampup_kimg       = 0,        # Duration of learning rate ramp-up.
    tick_kimg_base          = 1,        # Default interval of progress snapshots.
    tick_kimg_dict          = {}): # Resolution-specific overrides.

    # Initialize result dict.
    s = dnnlib.EasyDict()
    s.kimg = cur_nimg / 1000.0

    # Training phase.
    phase_dur = lod_training_kimg + lod_transition_kimg
    phase_idx = int(np.floor(s.kimg / phase_dur)) if phase_dur > 0 else 0
    phase_kimg = s.kimg - phase_idx * phase_dur

    # Level-of-detail and resolution.
    if lod_initial_resolution is None:
        s.lod = 0.0
    else:
        s.lod = training_set.resolution_log2
        s.lod -= np.floor(np.log2(lod_initial_resolution))
        s.lod -= phase_idx
        if lod_transition_kimg > 0:
            s.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
        s.lod = max(s.lod, 0.0)
    s.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(s.lod)))

    # Minibatch size.
    s.minibatch_size = minibatch_size_dict.get(s.resolution, minibatch_size_base)
    s.minibatch_gpu = minibatch_gpu_dict.get(s.resolution, minibatch_gpu_base)

    # Learning rate.
    s.G_lrate = G_lrate_dict.get(s.resolution, G_lrate_base)
    s.D_lrate = D_lrate_dict.get(s.resolution, D_lrate_base)
    if lrate_rampup_kimg > 0:
        rampup = min(s.kimg / lrate_rampup_kimg, 1.0)
        s.G_lrate *= rampup
        s.D_lrate *= rampup

    # Other parameters.
    s.tick_kimg = tick_kimg_dict.get(s.resolution, tick_kimg_base)
    return s

#----------------------------------------------------------------------------
# Main training script.

def training_loop(
    G_args                  = {},       # Options for generator network.
    D_args                  = {},       # Options for discriminator network.'
    G_opt_args              = {},       # Options for generator optimizer.
    D_opt_args              = {},       # Options for discriminator optimizer.
    G_loss_args             = {},       # Options for generator loss.
    D_loss_args             = {},       # Options for discriminator loss.
    dataset_args            = {},       # Options for dataset.load_dataset().
    sched_args              = {},       # Options for train.TrainingSchedule.
    grid_args               = {},       # Options for train.setup_snapshot_image_grid().
    metric_arg_list         = [],       # Options for MetricGroup.
    tf_config               = {},       # Options for tflib.init_tf().
    data_dir                = None,     # Directory to load datasets from.
    G_smoothing_kimg        = 10.0,     # Half-life of the running average of generator weights.
    minibatch_repeats       = 4,        # Number of minibatches to run before adjusting training parameters.
    lazy_regularization     = True,     # Perform regularization as a separate training step?
    G_reg_interval          = 4,        # How often the perform regularization for G? Ignored if lazy_regularization=False.
    D_reg_interval          = 16,       # How often the perform regularization for D? Ignored if lazy_regularization=False.
    reset_opt_for_new_lod   = True,     # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    mirror_augment          = False,    # Enable mirror augment?
    drange_net              = [-1,1],   # Dynamic range used when feeding image data to the networks.
    save_tf_graph           = False,    # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = False,    # Include weight histograms in the tfevents file?
    resume_pkl              = None,     # Network pickle to resume training from, None = train from scratch.

    data_size               = 3000,     # Number of images in the dataset
    num_epochs              = 10000,    # Number of epochs in total

    init_proj_dim           = None,     # Dimension of random projection for DCI NN search, None = no projection, 0 = estimated from sklearn
    init_staleness          = 10,       # Initial epoch multiplier to update NN candidates in IMLE (grows exponentially in the power of 2)
    num_samples_factor      = 25,       # The multiplier of number of NN candidates in IMLE w.r.t. data size
    knn_perturb_factor      = 0.1,      # Std of a normal perturbation added to the 1NN result in IMLE
    candidate_batch_size    = 256,      # Batch size for NN candidate feature calculation in IMLE
    exclusive_retrieved_code = 0,       # Enforce uniqueness of IMLE retrieved NN latent code?
    dist_thres_percentile   = 100.0,    # Percentile of NN pairs, based on latent reconstruction distances, to be considered for IMLE
    attr_interesting        = None,     # The interesting CelebA attributes of a minority subgroup, None = the entire dataset
    ):   

    # Initialize dnnlib and TensorFlow.
    tflib.init_tf(tf_config)
    num_gpus = dnnlib.submit_config.num_gpus
    image_snapshot_ticks = data_size//1000 # How often to save image snapshots? None = only save 'reals.png' and 'fakes-init.png'.
    network_snapshot_ticks = data_size//1000*5 # How often to save network snapshots? None = only save 'networks-final.pkl'.

    # Load training set.
    training_set = dataset.load_dataset(data_dir=dnnlib.convert_path(data_dir), shuffle_mb=0, verbose=True, **dataset_args)
    training_set_rec = dataset.load_dataset(data_dir=dnnlib.convert_path(data_dir), shuffle_mb=0, verbose=False, **dataset_args)
    grid_size, grid_reals, grid_labels = misc.setup_snapshot_image_grid(training_set, **grid_args)
    misc.save_image_grid(grid_reals, dnnlib.make_run_dir_path('arb-reals.png'), drange=training_set.dynamic_range, grid_size=grid_size)

    if attr_interesting is not None:
        attr_file = 'celeba/Anno/list_attr_celeba.txt'
        assert os.path.isfile(attr_file)
        print('Loading attributes from "%s"...' % attr_file)
        with open(attr_file) as f:
            lines = f.readlines()
        attr_names = lines[1].split()

    # Construct or load networks.
    with tf.device('/gpu:0'):
        if resume_pkl is None:
            print('Constructing networks...')
            resume_kimg = 0.0
            resume_time = 0.0
            G = tflib.Network('G', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **G_args)
            D = tflib.Network('D', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **D_args)
            Gs = G.clone('Gs')
        else:
            print('Loading networks from "%s"...' % resume_pkl)
            resume_kimg, resume_time = misc.resume_kimg_time(resume_pkl)
            G, D, Gs = misc.load_pkl(resume_pkl)
        lpips = misc.load_pkl('metrics/vgg16_zhang_perceptual.pkl')
        proj_dim = func_proj_dim(init_proj_dim, data_size, num_samples_factor, G)
        dci_db = DCI(proj_dim, num_comp_indices=3, num_simp_indices=15)

    # Print layers and generate initial image snapshot.
    G.print_layers(); D.print_layers()
    sched = training_schedule(cur_nimg=total_kimg*1000, training_set=training_set, **sched_args)
    grid_latents = np.random.randn(np.prod(grid_size), *G.input_shapes[0][1:])

    # Build random projector
    if init_proj_dim is not None:
        print('Building random projector %d to %d...' % (np.prod(G.output_shape[1:]), proj_dim))
        projector_path = 'random_projector_mat_%dto%d.npy' % (np.prod(G.output_shape[1:]), proj_dim)
        if os.path.isfile(projector_path):
            projector = np.load(projector_path)
        else:
            projector = np.random.normal(loc=0.0, scale=1.0/float(proj_dim), size=(np.prod(G.output_shape[1:]), proj_dim)).astype(np.float64)
            np.save(projector_path, projector)

    # Setup training inputs.
    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'), tf.device('/cpu:0'):
        lod_in               = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in             = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_size_in    = tf.placeholder(tf.int32, name='minibatch_size_in', shape=[])
        minibatch_gpu_in     = tf.placeholder(tf.int32, name='minibatch_gpu_in', shape=[])
        minibatch_multiplier = minibatch_size_in // (minibatch_gpu_in * num_gpus)
        Gs_beta              = 0.5 ** tf.div(tf.cast(minibatch_size_in, tf.float32), G_smoothing_kimg * 1000.0) if G_smoothing_kimg > 0.0 else 0.0

        sched = training_schedule(cur_nimg=int(resume_kimg*1000), training_set=training_set, **sched_args)
        reals_rec_1 = tf.placeholder(tf.float32, name='reals_rec_1', shape=[sched.minibatch_size]+training_set.shape)
        labels_rec_1 = tf.placeholder(tf.float32, name='labels_rec_1', shape=[sched.minibatch_size,training_set.label_size])
        latents_rec_1 = tf.placeholder(tf.float32, name='latents_rec_1', shape=[sched.minibatch_size]+G.input_shapes[0][1:])
        reals_rec_2 = tf.placeholder(tf.float32, name='reals_rec_2', shape=[sched.minibatch_size]+training_set.shape)
        labels_rec_2 = tf.placeholder(tf.float32, name='labels_rec_2', shape=[sched.minibatch_size,training_set.label_size])
        latents_rec_2 = tf.placeholder(tf.float32, name='latents_rec_2', shape=[sched.minibatch_size]+G.input_shapes[0][1:])
        reals, labels = training_set.get_minibatch_tf()
        reals_rec_1_split = tf.split(reals_rec_1, num_gpus)
        labels_rec_1_split = tf.split(labels_rec_1, num_gpus)
        latents_rec_1_split = tf.split(latents_rec_1, num_gpus)
        reals_rec_2_split = tf.split(reals_rec_2, num_gpus)
        labels_rec_2_split = tf.split(labels_rec_2, num_gpus)
        latents_rec_2_split = tf.split(latents_rec_2, num_gpus)
        reals_split = tf.split(reals, num_gpus)
        labels_split = tf.split(labels, num_gpus)

    # Setup optimizers.
    G_opt_args = dict(G_opt_args)
    D_opt_args = dict(D_opt_args)
    for args, reg_interval in [(G_opt_args, G_reg_interval), (D_opt_args, D_reg_interval)]:
        args['minibatch_multiplier'] = minibatch_multiplier
        args['learning_rate'] = lrate_in
        if lazy_regularization:
            mb_ratio = reg_interval / (reg_interval + 1)
            args['learning_rate'] *= mb_ratio
            if 'beta1' in args: args['beta1'] **= mb_ratio
            if 'beta2' in args: args['beta2'] **= mb_ratio
    G_opt = tflib.Optimizer(name='TrainG', **G_opt_args)
    D_opt = tflib.Optimizer(name='TrainD', **D_opt_args)
    G_reg_opt = tflib.Optimizer(name='RegG', share=G_opt, **G_opt_args)
    D_reg_opt = tflib.Optimizer(name='RegD', share=D_opt, **D_opt_args)

    # Build training graph for each GPU.
    for gpu in range(num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):

            # Create GPU-specific shadow copies of G and D.
            G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow')
            D_gpu = D if gpu == 0 else D.clone(D.name + '_shadow')
            lpips_gpu = lpips if gpu == 0 else lpips.clone(lpips.name + '_shadow')

            # Data alloation
            reals_rec_1_gpu, labels_rec_1_gpu = process_reals(reals_rec_1_split[gpu], labels_rec_1_split[gpu], lod_in, mirror_augment, training_set.dynamic_range, drange_net)
            latents_rec_1_gpu = latents_rec_1_split[gpu]
            reals_rec_2_gpu, labels_rec_2_gpu = process_reals(reals_rec_2_split[gpu], labels_rec_2_split[gpu], lod_in, mirror_augment, training_set.dynamic_range, drange_net)
            latents_rec_2_gpu = latents_rec_2_split[gpu]
            reals_gpu, labels_gpu = process_reals(reals_split[gpu], labels_split[gpu], lod_in, mirror_augment, training_set.dynamic_range, drange_net)

            # Evaluate loss functions.
            lod_assign_ops = []
            if 'lod' in G_gpu.vars: lod_assign_ops += [tf.assign(G_gpu.vars['lod'], lod_in)]
            if 'lod' in D_gpu.vars: lod_assign_ops += [tf.assign(D_gpu.vars['lod'], lod_in)]
            with tf.control_dependencies(lod_assign_ops):
                with tf.name_scope('G_loss'):
                    G_loss, G_reg = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, lpips=lpips_gpu, training_set=training_set, minibatch_size=minibatch_gpu_in, reals_rec_1=reals_rec_1_gpu, labels_rec_1=labels_rec_1_gpu, latents_rec_1=latents_rec_1_gpu, reals_rec_2=reals_rec_2_gpu, labels_rec_2=labels_rec_2_gpu, latents_rec_2=latents_rec_2_gpu, **G_loss_args)
                with tf.name_scope('D_loss'):
                    D_loss, D_reg = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, training_set=training_set, minibatch_size=minibatch_gpu_in, reals=reals_gpu, labels=labels_gpu, **D_loss_args)

            # Register gradients.
            if not lazy_regularization:
                if G_reg is not None: G_loss += G_reg
                if D_reg is not None: D_loss += D_reg
            else:
                if G_reg is not None: G_reg_opt.register_gradients(tf.reduce_mean(G_reg * G_reg_interval), G_gpu.trainables)
                if D_reg is not None: D_reg_opt.register_gradients(tf.reduce_mean(D_reg * D_reg_interval), D_gpu.trainables)
            G_opt.register_gradients(tf.reduce_mean(G_loss), G_gpu.trainables)
            D_opt.register_gradients(tf.reduce_mean(D_loss), D_gpu.trainables)

    # Setup training ops.
    G_train_op = G_opt.apply_updates()
    D_train_op = D_opt.apply_updates()
    G_reg_op = G_reg_opt.apply_updates(allow_no_op=True)
    D_reg_op = D_reg_opt.apply_updates(allow_no_op=True)
    Gs_update_op = Gs.setup_as_moving_average_of(G, beta=Gs_beta)

    # Finalize graph.
    with tf.device('/gpu:0'):
        try:
            peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
        except tf.errors.NotFoundError:
            peak_gpu_mem_op = tf.constant(0)
    tflib.init_uninitialized_vars()

    print('Initializing logs...')
    summary_log = tf.summary.FileWriter(dnnlib.make_run_dir_path())
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        G.setup_weight_histograms(); D.setup_weight_histograms()
    metrics = metric_base.MetricGroup(metric_arg_list)

    print('Training for %d kimg...\n' % total_kimg)
    dnnlib.RunContext.get().update('', cur_epoch=resume_kimg, max_epoch=total_kimg)
    maintenance_time = dnnlib.RunContext.get().get_last_update_interval()
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = -1
    tick_start_nimg = cur_nimg
    prev_lod = -1.0
    running_mb_counter = 0
    cursor = 0
    latent_candidates = np.random.randn(data_size*num_samples_factor, *G.input_shapes[0][1:]).astype(np.float32)

    selected_latents = None
    cur_reals_rec_double_select_remained = None
    cur_labels_rec_double_select_remained = None
    cur_latents_rec_double_select_remained = None
    tick_reals_rec_double_old = None
    while cur_nimg < total_kimg * 1000:
        if dnnlib.RunContext.get().should_stop(): break

        # Choose training parameters and configure training ops.
        sched = training_schedule(cur_nimg=cur_nimg, training_set=training_set, **sched_args)
        assert sched.minibatch_size % (sched.minibatch_gpu * num_gpus) == 0
        assert data_size % (sched.minibatch_size*2) == 0
        training_set.configure(sched.minibatch_size*2, sched.lod)
        training_set_rec.configure(sched.minibatch_size*2, sched.lod)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                G_opt.reset_optimizer_state(); D_opt.reset_optimizer_state()
        prev_lod = sched.lod

        # Run training ops.
        feed_dict = {lod_in: sched.lod, lrate_in: sched.G_lrate, minibatch_size_in: sched.minibatch_size, minibatch_gpu_in: sched.minibatch_gpu}
        for _repeat in range(minibatch_repeats):
            rounds = range(0, sched.minibatch_size, sched.minibatch_gpu * num_gpus)
            run_G_reg = (lazy_regularization and running_mb_counter % G_reg_interval == 0)
            run_D_reg = (lazy_regularization and running_mb_counter % D_reg_interval == 0)
            
            # Construct DCI
            if selected_latents is None or cur_nimg//(data_size*init_staleness) != (cur_nimg-sched.minibatch_size*2)//(data_size*init_staleness):
                if selected_latents is not None:
                    init_staleness *= 2
                label_candidates = training_set_rec.get_random_labels_np(data_size*num_samples_factor)
                proj_candidates = np.zeros((data_size*num_samples_factor, proj_dim)).astype(np.float64)
                for i in range((data_size*num_samples_factor)//candidate_batch_size+1):
                    print('\rCandidates sampling %d/%d...' % (i, (data_size*num_samples_factor)//candidate_batch_size), end='')
                    image_candidates = G.run(latent_candidates[i*candidate_batch_size:(i+1)*candidate_batch_size,:], label_candidates[i*candidate_batch_size:(i+1)*candidate_batch_size,:], is_validation=True, minibatch_size=sched.minibatch_size, num_gpus=min([2,num_gpus]))
                    if init_proj_dim is None:
                        proj_candidates[i*candidate_batch_size:(i+1)*candidate_batch_size,:] = np.reshape(image_candidates, (-1,np.prod(image_candidates.shape[1:]))).astype(np.float64)
                    else:
                        proj_candidates[i*candidate_batch_size:(i+1)*candidate_batch_size,:] = np.matmul(np.reshape(image_candidates, (-1,np.prod(image_candidates.shape[1:]))).astype(np.float64), projector)
                print('DCI constructing...')
                dci_db.reset()
                dci_db.add(proj_candidates, num_levels=3, field_of_view=10, prop_to_retrieve=0.002)
                del proj_candidates
                del label_candidates
                # Query DCI
                nearest_indices_list = []
                nearest_dists_list = []
                while len(nearest_indices_list) != data_size:
                    print('\rDCI querying %d/%d...' % (len(nearest_indices_list), data_size), end='')
                    cur_reals_double, cur_labels_double = training_set_rec.get_minibatch_np(sched.minibatch_size*2)
                    cur_reals_double = cur_reals_double.astype(np.float32)
                    if init_proj_dim is None:
                        cur_proj_double = np.reshape(misc.adjust_dynamic_range(cur_reals_double, training_set.dynamic_range, drange_net), (-1,np.prod(cur_reals_double.shape[1:]))).astype(np.float64)
                    else:
                        cur_proj_double = np.matmul(np.reshape(misc.adjust_dynamic_range(cur_reals_double, training_set.dynamic_range, drange_net), (-1,np.prod(cur_reals_double.shape[1:]))).astype(np.float64), projector)
                    if exclusive_retrieved_code:
                        nearest_indices_double, nearest_dists_double = dci_db.query(cur_proj_double, num_neighbours=num_samples_factor, field_of_view=200, prop_to_retrieve=1.0)
                        nearest_indices_double = np.array(nearest_indices_double)
                        nearest_dists_double = np.array(nearest_dists_double)
                        for i in range(sched.minibatch_size*2):
                            added = False
                            for j in range(num_samples_factor):
                                if nearest_indices_double[i,j] not in nearest_indices_list:
                                    nearest_indices_list.append(nearest_indices_double[i,j])
                                    nearest_dists_list.append(nearest_dists_double[i,j])
                                    added = True
                                    break
                            if not added:
                                nearest_indices_list.append(nearest_indices_double[i,0])
                                nearest_dists_list.append(nearest_dists_double[i,0])
                    else:
                        nearest_indices_double, nearest_dists_double = dci_db.query(cur_proj_double, num_neighbours=1, field_of_view=200, prop_to_retrieve=1.0)
                        nearest_indices_double = np.array(nearest_indices_double)
                        nearest_dists_double = np.array(nearest_dists_double)
                        nearest_indices_list += list(nearest_indices_double[:,0])
                        nearest_dists_list += list(nearest_dists_double[:,0])
                    cursor += sched.minibatch_size*2
                selected_latents = latent_candidates[np.array(nearest_indices_list)]
                selected_dists = np.array(nearest_dists_list)
                dist_thres = np.percentile(selected_dists, dist_thres_percentile)

            # Sync IMLE loss with G training loss
            cur_reals_rec_double = None if cur_reals_rec_double_select_remained is None or cursor % data_size == 0 else np.array(cur_reals_rec_double_select_remained)
            cur_labels_rec_double = None if cur_labels_rec_double_select_remained is None or cursor % data_size == 0 else np.array(cur_labels_rec_double_select_remained)
            cur_latents_rec_double = None if cur_latents_rec_double_select_remained is None or cursor % data_size == 0 else np.array(cur_latents_rec_double_select_remained)
            while cur_reals_rec_double is None or cur_reals_rec_double.shape[0] < sched.minibatch_size*2:
                cur_reals_rec_double_temp, cur_labels_rec_double_temp = training_set_rec.get_minibatch_np(sched.minibatch_size*2)
                cur_reals_rec_double_temp = cur_reals_rec_double_temp.astype(np.float32)
                cur_latents_rec_double_temp = selected_latents[(cursor%data_size):(cursor%data_size)+sched.minibatch_size*2]
                if attr_interesting is None:
                    selected_idx = selected_dists[(cursor%data_size):(cursor%data_size)+sched.minibatch_size*2] <= dist_thres
                else:
                    attr_interesting_list = attr_interesting.split(',')
                    active = np.ones(cur_labels_rec_double_temp.shape[0])
                    for attr in attr_interesting_list:
                        idx = attr_names.index(attr)
                        active *= cur_labels_rec_double_temp[:,idx]
                    selected_idx = active == 1
                cur_reals_rec_double_select = cur_reals_rec_double_temp[selected_idx]
                cur_labels_rec_double_select = cur_labels_rec_double_temp[selected_idx]
                cur_latents_rec_double_select = cur_latents_rec_double_temp[selected_idx]
                cur_reals_rec_double = np.array(cur_reals_rec_double_select) if cur_reals_rec_double is None or cursor % data_size == 0 else np.concatenate((cur_reals_rec_double, cur_reals_rec_double_select), axis=0)
                cur_labels_rec_double = np.array(cur_labels_rec_double_select) if cur_labels_rec_double is None or cursor % data_size == 0 else np.concatenate((cur_labels_rec_double, cur_labels_rec_double_select), axis=0)
                cur_latents_rec_double = np.array(cur_latents_rec_double_select) if cur_latents_rec_double is None or cursor % data_size == 0 else np.concatenate((cur_latents_rec_double, cur_latents_rec_double_select), axis=0)
                if cur_reals_rec_double.shape[0] > sched.minibatch_size*2:
                    cur_reals_rec_double_select_remained = np.array(cur_reals_rec_double[sched.minibatch_size*2:]); cur_reals_rec_double = np.array(cur_reals_rec_double[:sched.minibatch_size*2])
                    cur_labels_rec_double_select_remained = np.array(cur_labels_rec_double[sched.minibatch_size*2:]); cur_labels_rec_double = np.array(cur_labels_rec_double[:sched.minibatch_size*2])
                    cur_latents_rec_double_select_remained = np.array(cur_latents_rec_double[sched.minibatch_size*2:]); cur_latents_rec_double = np.array(cur_latents_rec_double[:sched.minibatch_size*2])
                else:
                    cur_reals_rec_double_select_remained = None
                    cur_labels_rec_double_select_remained = None
                    cur_latents_rec_double_select_remained = None
                if cursor % data_size == 0:
                    beginning = True
                cursor += sched.minibatch_size*2

            cur_reals_rec_1 = cur_reals_rec_double[:sched.minibatch_size]
            cur_labels_rec_1 = cur_labels_rec_double[:sched.minibatch_size]
            cur_reals_rec_2 = cur_reals_rec_double[sched.minibatch_size:]
            cur_labels_rec_2 = cur_labels_rec_double[sched.minibatch_size:]
            cur_latents_rec_double = misc.slerp(cur_latents_rec_double, np.random.randn(*cur_latents_rec_double.shape).astype(np.float32), knn_perturb_factor)
            cur_latents_rec_1 = cur_latents_rec_double[:sched.minibatch_size]
            cur_latents_rec_2 = cur_latents_rec_double[sched.minibatch_size:]
            if beginning:
                tick_reals_rec_double = np.array(cur_reals_rec_double)
                tick_labels_rec_double = np.array(cur_labels_rec_double)
                tick_latents_rec_double = np.array(cur_latents_rec_double)
                beginning = False

            order = np.arange(sched.minibatch_size)
            np.random.shuffle(order)
            cur_reals_rec_1 = cur_reals_rec_1[order]
            cur_labels_rec_1 = cur_labels_rec_1[order]
            cur_latents_rec_1 = cur_latents_rec_1[order]
            np.random.shuffle(order)
            cur_reals_rec_2 = cur_reals_rec_2[order]
            cur_labels_rec_2 = cur_labels_rec_2[order]
            cur_latents_rec_2 = cur_latents_rec_2[order]

            # Fast path without gradient accumulation.
            assert len(rounds) == 1
            feed_dict[reals_rec_1] = cur_reals_rec_1
            feed_dict[labels_rec_1] = cur_labels_rec_1
            feed_dict[latents_rec_1] = cur_latents_rec_1
            feed_dict[reals_rec_2] = cur_reals_rec_2
            feed_dict[labels_rec_2] = cur_labels_rec_2
            feed_dict[latents_rec_2] = cur_latents_rec_2
            tflib.run([G_train_op, G_loss], feed_dict)
            if run_G_reg:
                tflib.run(G_reg_op, feed_dict)
            tflib.run([D_train_op, D_loss, Gs_update_op], feed_dict)
            if run_D_reg:
                tflib.run(D_reg_op, feed_dict)

            cur_nimg += sched.minibatch_size*2
            running_mb_counter += 1

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_tick < 0 or cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
            cur_tick += 1
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = dnnlib.RunContext.get().get_time_since_last_update()
            total_time = dnnlib.RunContext.get().get_time_since_start() + resume_time

            # Report progress.
            print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %-6.1f gpumem %.1f' % (
                autosummary('Progress/tick', cur_tick),
                autosummary('Progress/kimg', cur_nimg / 1000.0),
                autosummary('Progress/lod', sched.lod),
                autosummary('Progress/minibatch', sched.minibatch_size),
                dnnlib.util.format_time(autosummary('Timing/total_sec', total_time)),
                autosummary('Timing/sec_per_tick', tick_time),
                autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                autosummary('Timing/maintenance_sec', maintenance_time),
                autosummary('Resources/peak_gpu_mem_gb', peak_gpu_mem_op.eval() / 2**30)))
            autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))

            # Save snapshots.
            if image_snapshot_ticks is not None and (cur_tick % image_snapshot_ticks == 0 or done):
                grid_fakes = Gs.run(grid_latents, grid_labels, is_validation=True, minibatch_size=sched.minibatch_gpu)
                misc.save_image_grid(grid_fakes, dnnlib.make_run_dir_path('arb-fakes-%06d.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size)
                if tick_reals_rec_double_old is None or np.sum(tick_reals_rec_double!=tick_reals_rec_double_old) > 0:
                    misc.save_image_grid(tick_reals_rec_double, dnnlib.make_run_dir_path('rec-reals.png'), drange=training_set.dynamic_range, grid_size=(8,(sched.minibatch_size*2)//8))
                    tick_reals_rec_double_old = np.array(tick_reals_rec_double)
                tick_fakes_nn = Gs.run(tick_latents_rec_double, tick_labels_rec_double, is_validation=True, minibatch_size=sched.minibatch_gpu)
                misc.save_image_grid(tick_fakes_nn, dnnlib.make_run_dir_path('rec-fakes-%06d.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=(8,(sched.minibatch_size*2)//8))
            if network_snapshot_ticks is not None and (cur_tick % network_snapshot_ticks == 0 or done):
                pkl = dnnlib.make_run_dir_path('network-snapshot-%06d.pkl' % (cur_nimg // 1000))
                misc.save_pkl((G, D, Gs), pkl)
                metrics.run(pkl, run_dir=dnnlib.make_run_dir_path(), data_dir=dnnlib.convert_path(data_dir), num_gpus=min([2,num_gpus]), tf_config=tf_config)

            # Update summaries and RunContext.
            metrics.update_autosummaries()
            tflib.autosummary.save_summaries(summary_log, cur_nimg)
            dnnlib.RunContext.get().update('%.2f' % sched.lod, cur_epoch=cur_nimg // 1000, max_epoch=total_kimg)
            maintenance_time = dnnlib.RunContext.get().get_last_update_interval() - tick_time

    # Save final snapshot.
    misc.save_image_grid(grid_fakes, dnnlib.make_run_dir_path('arb-fakes-final.png'), drange=drange_net, grid_size=grid_size)
    misc.save_image_grid(tick_fakes_nn, dnnlib.make_run_dir_path('rec-fakes-final.png'), drange=drange_net, grid_size=(8,(sched.minibatch_size*2)//8))
    misc.save_pkl((G, D, Gs), dnnlib.make_run_dir_path('network-final.pkl'))

    # All done.
    summary_log.close()
    training_set.close()
    training_set_rec.close()

#----------------------------------------------------------------------------
