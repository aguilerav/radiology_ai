# Importing required libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tf.keras.backend.clear_session()
from tensorflow.keras.layers import Input, Softmax, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Downloading the Chest X-ray dataset
#!wget https://www.dropbox.com/s/73s9n7nugqrv1h7/Dataset.zip?dl=1 -O 'archive.zip'
# Unzipping the dataset and delete the .zip file
#!unzip -q '/content/archive.zip'
#!rm -rf '/content/archive.zip'

# Settting up batch size, random seed, and the dataset path

TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 16
SEED = 21
dataset_path = '/content/Dataset'

# Initialising ImageDataGenerator for data augmentation
# We use random horizontal flip for augmentation
# Pixels will be notmalised between 0 and 1
  # zca_epsilon: Epsilon for ZCA whitening. Default is 1e-6
  # Horizontal_flip: Boolean. Randomly flip inputs horizontally.
  # Rescale: Rescaling factor, defaults to None.
           # If None or 0, no rescaling is applied, otherwise it multiplied the data by the value provided
           # (after applying all other transformations)

train_val_gen = ImageDataGenerator(zca_epsilon = 0.0,
                                   horizontal_flip = True,
                                   rescale = 1./255)        # Do not change rescale

test_gen = ImageDataGenerator(zca_epsilon = 0.0,
                              horizontal_flip = False,
                              rescale = 1./255)             # Do not change rescale

# Taking input of the train, validation, and test images using flow_from_directory() function
# Setting the image size to (224, 224) and setting the batch size

train_datagen = train_val_gen.flow_from_directory(directory = dataset_path + '/train',
                                                  target_size = (224, 224),
                                                  color_mode = "rgb",
                                                  classes = None,
                                                  class_mode = "categorical",
                                                  batch_size = TRAIN_BATCH_SIZE,
                                                  shuffle = True,
                                                  seed = SEED,
                                                  interpolation = "nearest")

val_datagen = train_val_gen.flow_from_directory(directory = dataset_path + '/val',
                                                target_size = (224, 224),
                                                color_mode = "rgb",
                                                classes = None,
                                                class_mode = "categorical",
                                                batch_size = VAL_BATCH_SIZE,
                                                shuffle = True,
                                                seed = SEED,
                                                interpolation = "nearest")


# For testing, we should take one input at a time. Hence, batch_size = 1

test_datagen = test_gen.flow_from_directory(directory = dataset_path + '/test',
                                            target_size = (224, 224),
                                            color_mode = "rgb",
                                            classes = None,
                                            class_mode = "categorical",
                                            batch_size = 1,
                                            shuffle = False,
                                            seed = SEED,
                                            interpolation = "nearest")

# Verificar el mapeo de clases
print("Clases de entrenamiento:", train_datagen.class_indices)
print("Clases de validación:", val_datagen.class_indices)
print("Clases de prueba:", test_datagen.class_indices)

# Initialising pretrained model and passing the imagenet weights
pretrained_model = tf.keras.applications.EfficientNetB0(weights='imagenet',
                                                        include_top=False,
                                                        input_shape=(224, 224, 3),
                                                        pooling = 'max')

# Freeze pre trained layers
pretrained_model.trainable = False

# Prediction layers
x = pretrained_model.output
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)  # Capa fully connected con regularización
x = Dropout(0.5)(x)  # Dropout para evitar sobreajuste
predictions = Dense(3, activation='softmax')(x)

# Defining new model's input and output layers
model = Model(inputs=pretrained_model.input, outputs=predictions)

# We use the SGD optimiser, with a very low learning rate, and loss function which is specific to two class classification
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# You can directly save the model into your Google drive by changing the below path
model_filepath = '/content/drive/MyDrive/models/best_model.keras'


# ModelCheckpoint callback will save models weight if the training accuracy of the model has increased from the previous epoch
model_save = tf.keras.callbacks.ModelCheckpoint(model_filepath,
                                                monitor = "val_accuracy",
                                                verbose = 1,
                                                save_best_only = True,
                                                save_weights_only = False,
                                                mode = "max",
                                                save_freq = "epoch")
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

callbacks = [model_save, early_stopping, reduce_lr]

history = model.fit(train_datagen,
                    epochs = 20,
                    steps_per_epoch = (len(train_datagen)),
                    validation_data = val_datagen,
                    # validation_steps = (len(val_datagen)),
                    shuffle = False,
                    callbacks = callbacks)

