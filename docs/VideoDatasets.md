# Turning video frame data into usable datasets for model training

--

I began to research how to turn video frame data into datasets or objects that can be used to train our model on more motion based gestures such as complete words or specific letters like J and Z. At first I thought the process would be similar to how image data directories can be converted into tensor flow datasets but as I continued to research I found that the process is much more complicated. 

The Mendeley ASL dataset that I have been using to extract image data of ASL gestures also includes a diretory containing subdirectories for the alphabet, that contain individual letter files which have video files of someone demonstrating the gesture for the respective letters along with a folder alongside it which has pngs for each frame of those videos. Each letter has multiple clips and multiple frame_pngs folders for those clips. It simplifies the process of having to splice those specific clips which can help to train the model. 

From certain sources online, I would have to create labels for each of the clip_png folders that are in all of the letters then put their frame data into something like Numpy arrays that can be used to train the model. However some of the research has been a little difficult as many sources handle how to convert actual videos into these datasets rather than starting with the frame data. 

---

Helpful YouTube video source: https://www.youtube.com/watch?v=3xualF8abC4&t=45s


TensorFlow site source for loading video data: https://www.tensorflow.org/tutorials/load_data/video








