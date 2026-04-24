## Gesture video data extraction outline + information found 

--

This is documentation for my development process for the video frame data extraction function for building datasets. Im utilizing both mediapipe and opencv in order to scan over the image files containing video frames of ASL gestures being performed and add landmarks to better read the details of hand positions when doing ASL. 


## How it works

With how my local directory for the image/video datasets is set up, the file will sort through the "ASL_dynamic" folder for the specific video frame folders. From there it will iterate through each of the video frame pngs, scanning for hand gestures in-frame, in order to draw landmarks on them to better analyze the details of the hand gestures for the training model. 

Marking specific points on hands through multiple frames, similar to how Gabriel has the camera detection landmarks set up for the model right now, can help to better train the model on more specific examples of each letter and potentially full word gestures. The process so far seems somewhat similar to Gabriels work for drawing landmarks in the real time translation testing on a webcam, but varies in certain parts because its being applied to a static image. 

The environment and interpreter im using while making this is seperate from my TensorFlow environment and only has packages relating to mediapipe and opencv since there were still complications with hosting the those along with TensorFlow + other packages in the same environment on my local computer. There could be other ways around that but it was the quickest solution on my device and could possible help Dillon or Josh if needed. 

--

Ive been programming this alongside doing further research on other useful methods to help out both this function and other parts of theproject as well. While researching, I found a vscode extension for a TensorFlow tool called "TensorBoard" which is a tool built into TensorFlow that visualizes the data/learning progress of an LM its applied to. It can output the models accuracy ratio, log each training batch the model completes, and also visualizations like scalars, graphs etc.  

I put the link to the TensorBoard page on the TensorFlow site below. I think it could be useful for better observing the models training. 

## Tensorboard Link: 
https://www.tensorflow.org/tensorboard/get_started

