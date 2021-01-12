# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from training import misc

#----------------------------------------------------------------------------

class Projector:
    def __init__(self):
        self.num_steps                  = 1000
        self.initial_learning_rate      = 0.1
        self.initial_noise_factor       = 0.05
        self.lr_rampdown_length         = 0.25
        self.lr_rampup_length           = 0.05
        self.noise_ramp_length          = 0.75
        self.verbose                    = False
        self.clone_net                  = True

        self._Gs                    = None
        self._minibatch_size        = None
        self._latents_var           = None
        self._noise_in              = None
        self._latents_expr          = None
        self._images_expr           = None
        self._target_images_var     = None
        self._lpips                 = None
        self._dist                  = None
        self._loss                  = None
        self._lrate_in              = None
        self._opt                   = None
        self._opt_step              = None
        self._cur_step              = None

    def _info(self, *args):
        if self.verbose:
            print('Projector:', *args)

    def set_network(self, Gs, minibatch_size=1, num_steps=1000, initial_noise_factor=0.05):
        self._Gs = Gs
        self._minibatch_size = minibatch_size
        self.num_steps = num_steps
        self.initial_noise_factor = initial_noise_factor
        if self._Gs is None:
            return
        if self.clone_net:
            self._Gs = self._Gs.clone()

        # Image output graph.
        self._info('Building image output graph...')
        self._latents_var = tf.Variable(tf.zeros([self._minibatch_size] + list(self._Gs.input_shapes[0][1:])), name='latents_var')
        self._noise_in = tf.placeholder(tf.float32, [], name='noise_in')
        latents_noise = tf.random.normal(shape=self._latents_var.shape)
        self._latents_expr = tflib.slerp(self._latents_var, latents_noise, self._noise_in)
        try:
            self._images_expr = self._Gs.get_output_for(self._latents_expr, tf.zeros([self._minibatch_size] + list(self._Gs.input_shapes[1][1:])), is_validation=True)
        except:
            self._images_expr = self._Gs.get_output_for(self._latents_expr, is_validation=True)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        proc_images_expr = (self._images_expr + 1) * (255 / 2)
        sh = proc_images_expr.shape.as_list()
        if sh[2] > 256:
            factor = sh[2] // 256
            proc_images_expr = tf.reduce_mean(tf.reshape(proc_images_expr, [-1, sh[1], sh[2] // factor, factor, sh[2] // factor, factor]), axis=[3,5])

        # Loss graph.
        self._info('Building loss graph...')
        self._target_images_var = tf.Variable(tf.zeros(proc_images_expr.shape), name='target_images_var')
        if self._lpips is None:
            self._lpips = misc.load_pkl('metrics/vgg16_zhang_perceptual.pkl') # vgg16_zhang_perceptual.pkl
        self._dist = self._lpips.get_output_for(proc_images_expr, self._target_images_var)
        self._loss = tf.reduce_sum(self._dist)

        # Optimizer.
        self._info('Setting up optimizer...')
        self._lrate_in = tf.placeholder(tf.float32, [], name='lrate_in')
        self._opt = dnnlib.tflib.Optimizer(learning_rate=self._lrate_in)
        self._opt.register_gradients(self._loss, [self._latents_var])
        self._opt_step = self._opt.apply_updates()

    def run(self, target_images):
        # Run to completion.
        self.start(target_images)
        while self._cur_step < self.num_steps:
            self.step()

        # Collect results.
        pres = dnnlib.EasyDict()
        pres.latents = self.get_latents()
        pres.images = self.get_images()
        return pres

    def start(self, target_images, init_latents=None):
        assert self._Gs is not None

        # Prepare target images.
        self._info('Preparing target images...')
        target_images = np.asarray(target_images, dtype='float32')
        target_images = (target_images + 1) * (255 / 2)
        sh = target_images.shape
        assert sh[0] == self._minibatch_size
        if sh[2] > self._target_images_var.shape[2]:
            factor = sh[2] // self._target_images_var.shape[2]
            target_images = np.reshape(target_images, [-1, sh[1], sh[2] // factor, factor, sh[3] // factor, factor]).mean((3, 5))

        # Initialize optimization state.
        self._info('Initializing optimization state...')
        if init_latents is None:
            tflib.set_vars({self._target_images_var: target_images, self._latents_var: np.random.randn(self._minibatch_size, *self._Gs.input_shapes[0][1:])})
        else:
            tflib.set_vars({self._target_images_var: target_images, self._latents_var: init_latents})
        self._opt.reset_optimizer_state()
        self._cur_step = 0

    def step(self):
        assert self._cur_step is not None
        if self._cur_step >= self.num_steps:
            return
        if self._cur_step == 0:
            self._info('Running...')

        # Hyperparameters.
        t = self._cur_step / self.num_steps
        noise_strength = self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        learning_rate = self.initial_learning_rate * lr_ramp

        # Train.
        feed_dict = {self._noise_in: noise_strength, self._lrate_in: learning_rate}
        _, dist_value, loss_value = tflib.run([self._opt_step, self._dist, self._loss], feed_dict)

        # Print status.
        self._cur_step += 1
        if self._cur_step == self.num_steps or self._cur_step % 10 == 0:
            self._info('%-8d%-12g%-12g' % (self._cur_step, np.mean(dist_value), np.mean(loss_value)))
        if self._cur_step == self.num_steps:
            self._info('Done.')

    def get_cur_step(self):
        return self._cur_step

    def get_latents(self):
        return tflib.run(self._latents_expr, {self._noise_in: 0})

    def get_images(self):
        return tflib.run(self._images_expr, {self._noise_in: 0})

    def get_dist(self):
        return tflib.run(self._dist, {self._noise_in: 0})

    def get_loss(self):
        return tflib.run(self._loss, {self._noise_in: 0})

#----------------------------------------------------------------------------
