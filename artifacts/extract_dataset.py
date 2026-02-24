import tensorflow as tf
from tensorflow import keras
from pathlib import Path

import numpy as np

data_Dir = Path("data/raw_images/asl_alphabet")

 # converts the A and B ASL gesture image folders into their own datasets
training_DS, val_DS = keras.utils.image_dataset_from_directory(
    str(data_Dir),
    labels= 'inferred',
    batch_size = 128,
    validation_split = 0.3,
    subset = "both",
    seed = 1007
)

# store the label names for gesture files
class_names = training_DS.class_names

# Shuffling the batches in the dataset for randomness
training_DS = training_DS.shuffle(
    buffer_size = 1400,
    reshuffle_each_iteration = True
)

# report for what data is extracted
for images, labels in training_DS.take(1):
    label_ids = labels.numpy()
    label_names = [class_names[i] for i in label_ids[:10]]
    print(f"""    Total Classes: {len(class_names)}
    Class Names: {class_names}\n
    Training dataset: 
        {len(training_DS) * 128} Images
        {len(training_DS)} Batches\n  
    Batch Dimensions: {images.shape}
    First 10 image labels in a sample batch: {label_names}""")


batch_num = 1
# prints sample batches specifying their contained images and shapes
"""for images, labels in training_DS.take(5):
    label_ids = labels.numpy()
    label_names = [class_names[i] for i in label_ids[:10]]
    print("Batch: {}".format(batch_num))
    print("First 5 images: {}".format(label_names))
    print("Images: {}".format(images.shape))
    batch_num += 1 """




