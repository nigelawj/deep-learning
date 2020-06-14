# Convolutional Neural Network
#
# Dataset must be downloaded and split into training/test directories

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

def create_model():
    # Initialising the CNN
    classifier = Sequential()

    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection
    classifier.add(Dense(units=128, activation='relu'))
    # If non-binary outcome (> 2 categories) - use softmax function
    classifier.add(Dense(units=1, activation='sigmoid'))

    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier

# Part 2 - Fitting the CNN to the images
# NOTE: Dataset is not included
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('./dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('./dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

import os

# Prepare a directory to store all the checkpoints.
checkpoint_dir = '.\ckpt' # place sample checkpoint in this folder if needed
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Either restore the latest model, or create a fresh one
# if there is no checkpoint available.
checkpoints = [(checkpoint_dir + '\\' + name) for name in os.listdir(checkpoint_dir)]

if checkpoints:
    latest_checkpoint = max(checkpoints, key=os.path.getctime)

    print('Restoring from', latest_checkpoint)
    classifier = load_model(latest_checkpoint)
else:
    classifier = create_model()

    callbacks = [
        # This callback saves a SavedModel every 100 batches.
        # We include the training loss in the folder name.
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'ckpt-loss={loss:.2f}'),
            save_weights_only=True
            #save_best_only=True
        )
    ]

    classifier.fit(
        training_set,
        epochs=25,
        validation_data=test_set,
        callbacks=callbacks
    )

# Part 3 - Making new predictions
import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
# classifier takes batch input; must add new dimension
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices
if (result[0][0] == 1):
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction) # dog

test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
# classifier takes batch input; must add new dimension
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices
if (result[0][0] == 1):
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction) # cat!