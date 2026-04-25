## I was able to implement the methods used for generating individual frames that can be processed in the backend and fed to the ML model for understanding. Unfortunately I am unable to fully test this functionality as the detection implementation is starting in future sprints. 

# As of right now there are multiple methods added to the firebase.py file which allow the server to use firebase storage in order to store these frames temporarily while prcoessing such that we can avoid security and storage issues that may be caused by storing the images forever.

# I ended up having to edit the frontend file, dashboard, in order to implement some of the changes I wanted. This also made the two divs centered as well.