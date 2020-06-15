# deep-learning
Some simple starter .py and .ipynb codes for referencing DL basics
- Artificial Neural Networks
- Convolutional Neural Networks
- Recurrent Neural Networks (LSTM)
- Self Organising Maps
- Boltzmann Machines (Restricted)
- Autoencoders (Stacked)

# Notes
Keras is now bundled with tensorflow and no longer supports other backends like Theano;
- instead of `import keras` and `from keras.layers import Conv2D` it is recommended to do `from tensorflow.keras.layers import Conv2D`, unless a different backend is required; Note that `keras 2.3.0` would be the latest version supporting these other backends as per [keras github](https://github.com/keras-team/keras#multi-backend-keras-and-tfkeras)
- Implemented saving/loading of model checkpoints for CNN
	- To restart from 1st epoch or from last epoch would depend on requirements; not always sensible/possible to resume from last epoch
	- save_freq in ModelCheckpoint may not always be epoch, especially if epochs are quick to complete; may want to do it per fold in cross validation for e.g.

# Future Scope
- Codes could be optimised to increase GPU utilisation
- Hyperparameter tuning could utilise Keras Tuner instead of scikit-learn