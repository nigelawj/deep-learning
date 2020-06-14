# deep-learning
Some simple starter codes for referencing DL basics

# NOTE
Keras is now bundled with tensorflow and no longer supports other backends like Theano;
- instead of `import keras` and `from keras.layers import Conv2D` it is recommended to do `from tensorflow.keras.layers import Conv2D`, unless a different backend is required; Note that `keras 2.3.0` would be the latest version supporting these other backends as per [keras github](https://github.com/keras-team/keras#multi-backend-keras-and-tfkeras)

# Improvements
- Codes could be optimised to increase GPU utilisation
- Hyperparameter tuning could utilise Keras Tuner instead
- Implement saving of model checkpoints