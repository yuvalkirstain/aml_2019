# -*- coding: utf-8 -*-
"""AML-HW5-Section-F.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1c0iBr-Anu90XkAMjePxG2YSShRg240ui
"""

'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(0 / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# instantiate VAE model
vae = Model(x, x_decoded_mean)

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + 0 - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()


# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

"""Here we start writing our code

Section C - Add an encoder which
maps MNIST digits to the latent space. Using this encoder, visualize
the test set in the latent space
"""

# encoder which maps MNIST digits to the latent space
encoder = Model(x, z_mean)

# visualize the test set in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.show()

"""Section C - Take one image per digit and print its
corresponding mapping coordinates in the latent space, present the
answer as a table.
"""

# Take one image per digit and print its corresponding mapping coordinates
# in the latent space, present the answer as a table
NUM_OF_DIGITS = 10
digits_to_latent = {}
for i, digit in enumerate(y_test):
    if len(digits_to_latent) == NUM_OF_DIGITS:
        break
    if digit in digits_to_latent:
        continue
    digits_to_latent[digit] = (x_test_encoded[i, 0], x_test_encoded[i, 1])
assert len(digits_to_latent) == NUM_OF_DIGITS
plt.scatter([latent[0] for latent in digits_to_latent.values()], [latent[1] for latent in digits_to_latent.values()], c=list(digits_to_latent.keys()))
plt.table(colLabels=('latent vector', 'digit'),
                cellText=[[latent, digit] for digit, latent in digits_to_latent.items()])
plt.show()

"""Section D - Use the following code to define a generator that based on a sample from the latent space, generates a digits"""

# Use the following code to define a generator that based on a sample from
# the latent space, generates a digits.
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(inputs=decoder_input, outputs=_x_decoded_mean)
z_sample = np.array([[0.5, 0.2]])
x_decoded = generator.predict(z_sample)

plt.imshow(x_decoded.reshape(28,28))

"""Section E - Take two original images from MNIST of different digits.  Sample 10 points from the line connecting the two representations
in the latent space and generate their images
"""

# Take two original images from MNIST of different digits
FIRST_DIGIT = 0
SECOND_DIGIT = 9
first_z = digits_to_latent[FIRST_DIGIT]
second_z = digits_to_latent[SECOND_DIGIT]

# Sample 10 points from the line connecting the two representations
# in the latent space and generate their images
SAMPLE_AMOUNT = 10
sampled = list(zip(np.linspace(first_z[0], second_z[0], SAMPLE_AMOUNT), np.linspace(first_z[1], second_z[1], SAMPLE_AMOUNT)))
fig=plt.figure(figsize=(28, 28))
columns = 1
rows = 10
for i, z_sample in enumerate(sampled):
  x_sample = generator.predict(np.array([list(z_sample)]))
  fig.add_subplot(rows, columns, i+1)
  plt.imshow(x_sample.reshape(28,28))
plt.show()

