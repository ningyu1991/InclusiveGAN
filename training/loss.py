# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Loss functions."""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

#----------------------------------------------------------------------------
# Non-saturating logistic loss with path length regularizer from the paper
# "Analyzing and Improving the Image Quality of StyleGAN", Karras et al. 2019
# Plus reconstruction loss from IMLE, interplation loss composed of perceptual loss (from the closer latent code) and gram matrix loss (from the farther latent code), and adv loss of arbitrary latent code

def G_logistic_ns_rec_interp_arb_pathreg(G, D, lpips, training_set, minibatch_size, reals_rec_1, labels_rec_1, latents_rec_1, reals_rec_2, labels_rec_2, latents_rec_2,
    NN_rec_lpips_weight,
    pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0):

    loss =  None

    rec_images_1_out = G.get_output_for(latents_rec_1, labels_rec_1, is_training=True)
    rec_images_2_out = G.get_output_for(latents_rec_2, labels_rec_2, is_training=True)
    rec_images_1_out = (rec_images_1_out + 1) * (255 / 2)
    rec_images_2_out = (rec_images_2_out + 1) * (255 / 2)
    reals_rec_1 = (reals_rec_1 + 1) * (255 / 2)
    reals_rec_2 = (reals_rec_2 + 1) * (255 / 2)
    loss_NN_rec_lpips = (lpips.get_output_for(rec_images_1_out, reals_rec_1) + lpips.get_output_for(rec_images_2_out, reals_rec_2)) * 0.5
    loss_NN_rec_lpips *= NN_rec_lpips_weight
    loss_NN_rec_lpips = autosummary('Loss/loss_NN_rec_lpips', loss_NN_rec_lpips)
    loss = loss_addup(loss, loss_NN_rec_lpips)

    interp_factors = tf.random_uniform([minibatch_size,1], minval=0.0, maxval=1.0)
    interp_latents = tflib.slerp(latents_rec_2, latents_rec_1, interp_factors)
    interp_labels = tflib.lerp(labels_rec_2, labels_rec_1, interp_factors)
    interp_images_out = G.get_output_for(interp_latents, interp_labels, is_training=True)
    interp_images_out = (interp_images_out + 1) * (255 / 2)
    loss_NN_interp_lpips = tflib.lerp(lpips.get_output_for(interp_images_out, reals_rec_2), lpips.get_output_for(interp_images_out, reals_rec_1), tf.squeeze(interp_factors, axis=[1]))
    loss_NN_interp_lpips *= (NN_rec_lpips_weight*0.4)
    loss_NN_interp_lpips = autosummary('Loss/loss_NN_interp_lpips', loss_NN_interp_lpips)
    loss = loss_addup(loss, loss_NN_interp_lpips)

    latents_random = tf.random_normal([minibatch_size]+G.input_shapes[0][1:])
    labels_random = training_set.get_random_labels_tf(minibatch_size)
    arb_images_out = G.get_output_for(latents_random, labels_random, is_training=True)
    arb_scores_out, _ = D.get_output_for(arb_images_out, labels_random, is_training=True)
    loss_G_arb = tf.nn.softplus(-arb_scores_out)
    loss_G_arb = autosummary('Loss/loss_G_arb', loss_G_arb)
    loss = loss_addup(loss, loss_G_arb)

    # Path length regularization.
    with tf.name_scope('PathReg'):

        # Evaluate the regularization term using a smaller minibatch to conserve memory.
        pl_minibatch = minibatch_size // pl_minibatch_shrink
        pl_latents = tf.random_normal([pl_minibatch] + G.input_shapes[0][1:])
        pl_labels = training_set.get_random_labels_tf(pl_minibatch)
        fake_images_out, fake_dlatents_out = G.get_output_for(pl_latents, pl_labels, is_training=True, return_dlatents=True)

        # Compute |J*y|.
        pl_noise = tf.random_normal(tf.shape(fake_images_out)) / np.sqrt(np.prod(G.output_shape[2:]))
        pl_grads = tf.gradients(tf.reduce_sum(fake_images_out * pl_noise), [fake_dlatents_out])[0]
        pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))

        # Track exponential moving average of |J*y|.
        with tf.control_dependencies(None):
            pl_mean_var = tf.Variable(name='pl_mean', trainable=False, initial_value=0.0, dtype=tf.float32)
        pl_mean = pl_mean_var + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean_var)
        pl_update = tf.assign(pl_mean_var, pl_mean)

        # Calculate (|J*y|-a)^2.
        with tf.control_dependencies([pl_update]):
            pl_penalty = tf.square(pl_lengths - pl_mean)

        # Apply weight.
        #
        # Note: The division in pl_noise decreases the weight by num_pixels, and the reduce_mean
        # in pl_lengths decreases it by num_affine_layers. The effective weight then becomes:
        #
        # gamma_pl = pl_weight / num_pixels / num_affine_layers
        # = 2 / (r^2) / (log2(r) * 2 - 2)
        # = 1 / (r^2 * (log2(r) - 1))
        # = ln(2) / (r^2 * (ln(r) - ln(2))
        #
        reg = pl_penalty * pl_weight
        reg = autosummary('Loss/pl_penalty', reg)

    return loss, reg

def D_logistic_r1(G, D, training_set, minibatch_size, reals, labels,
    gamma=10.0):

    loss = None

    latents_random = tf.random_normal([minibatch_size*2]+G.input_shapes[0][1:])
    labels_random = training_set.get_random_labels_tf(minibatch_size*2)
    arb_images_out = G.get_output_for(latents_random, labels_random, is_training=True)
    arb_scores_out, _ = D.get_output_for(arb_images_out, labels_random, is_training=True)
    real_scores_out, _ = D.get_output_for(reals, labels, is_training=True)
    loss_D = tf.nn.softplus(arb_scores_out) + tf.nn.softplus(-real_scores_out)
    loss_D = autosummary('Loss/loss_D', loss_D)
    loss = loss_addup(loss, loss_D)    

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        reg = gradient_penalty * (gamma * 0.5)
        reg = autosummary('Loss/gradient_penalty_D', reg)
        
    return loss, reg

#----------------------------------------------------------------------------

def loss_addup(loss, loss_):
    if loss is None:
        L = tf.identity(loss_)
    else:
        L = loss + loss_
    return L