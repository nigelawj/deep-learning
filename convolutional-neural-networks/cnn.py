# Convolutional Neural Network - predict binary target (cat or dog)
#
# Dataset must be downloaded and split into training/test directories

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

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
from pathlib import Path

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    Path('./dataset/training_set'),
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    Path('./dataset/test_set'),
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

import re, os

# Prepare a directory to store all the checkpoints.
checkpoint_dir = Path('./checkpoints') # place sample checkpoint in this folder if needed
checkpoint_dir.mkdir(exist_ok=True, parents=True)

# Either restore the latest model, or create a fresh one
# if there is no checkpoint available.
checkpoints = [x for x in checkpoint_dir.iterdir() if x.is_dir()]

if (checkpoints):
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    latest_checkpoint = str(latest_checkpoint) # need to convert into str format

    print('Restoring from', latest_checkpoint)
    try:
        classifier = load_model(latest_checkpoint)
    except:
        print('Unable to load model. Check your model checkpoints folder!')
        sys.exit(1)

    latest_epoch = int(re.search('epoch=(\d*)-', latest_checkpoint).group(1))
else:
    classifier = create_model()
    latest_epoch = 0

# Comment this block out if not repeating training of model
###
callbacks = [
    EarlyStopping(patience=2), # stops and saves modelcheckpoint if no improvement in monitored quantity after 2 epochs
    ModelCheckpoint(
        # embed loss in checkpoint name instead of val_loss upon KeyboardInterrupt as val_loss is not available in between epochs; save_best_only still saves by best (min) val_loss
        filepath=str(Path(checkpoint_dir, 'epoch={epoch}-loss={loss:.2f}')),
        save_best_only=True,
        monitor='val_loss', # default
        mode='auto', # default
        save_freq='epoch', # default
        # period=2, # option to save every k epochs
    )
]

classifier.fit(
    training_set,
    epochs=25,
    validation_data=test_set,
    initial_epoch=latest_epoch, # resume from latest epoch
    callbacks=callbacks
)
###

# Part 3 - Making new predictions
import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img(Path('dataset/single_prediction/cat_or_dog_1.jpg'), target_size=(64, 64))
test_image = image.img_to_array(test_image)
# classifier takes batch input; must add new dimension
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)

print(training_set.class_indices)
if (result[0][0] == 1):
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction) # dog!

test_image2 = image.load_img(Path('dataset/single_prediction/cat_or_dog_2.jpg'), target_size=(64, 64))
test_image2 = image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis=0)
result = classifier.predict(test_image2)

print(training_set.class_indices)
if (result[0][0] == 1):
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction) # cat!