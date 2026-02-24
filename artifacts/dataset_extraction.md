# Created the extract_dataset file to take dataset information to use for model training

-- 

# I found that keras contains a built-in function that automatically takes images from a directory and turns them into their own datasets based on its structure with image_dataset_from_directory() which simplified a lot of the process

The datasets created from this function are returned as  tf.data.Dataset(format="tf") objects. With other TensorFlow functions, the tf dataset objects can be converted into numpy arrays or be fed directly to the model. The functionality of the current file just has them as tf datasets but when the model becomes more fleshed out to train from images, I will convert them into the prerferred format. 

My file currently utilizes the ASL gesture images from the alphabets A and B folders, from the Mendeley ASL dataset, and converts a portion of their images into respective datasets. There is a general report that prints when the file is run containing the attributes of the datasets including their classes, number of images, number of batches, etc. I did not incorporate the full alphabet dataset folders since I was only testing the functionality of my code. 

The extract_dataset.py file along with the data directory that stores the image data will be pushed to the git for documenting and group use for members who want to test the file with the same dataset portion I tested on. The full Mendeley Dataset will not be included as to not take up a large amount of space in the git and since the datasets themselves should mostly be used and stored locally. 

Keras documentation containing the built-in functions I utilized: https://keras.io/api/data_loading/image/


