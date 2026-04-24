# Methods for storing and displaying previous user translations 

-- 
# Goal

The translation page should display completed translations in the textbox area instead of only showing the current prediction near the users camera feed. The user should be able to see previous translations from the current session, in the text section of the page, and edit the text if needed. 

For the intial version of this, one of the more safer options seems to be to add in a button or toggle that indiciates when the user wants to store the translation, just for the sake of testing and efficiency. Later on if there is time to fully flesh out/actually implement this feature, we can change it to where the text storing aspect is automatic using some kind of confirmation logic or a similar approach but the whole group would have to discuss it. 

# Approach 

For implementation ill most liekly have to add in the code to the frontend on my own device after setting up the whole project locally. My approach as of now is to have the frontend store the current gesture translation in a state then have some kind of "add translation" action that will append the current translation into the textbox area of the translation page. Having the translations be stored and appended to the textbox for everyframe from the camerafeed in real time can lead to a large amount of repeated text being displayed. I can look through Dillons code for the frontend to see where I can start to approach the implementation. 

# Main three ways of indicating when translations should be stored and displayed: 

 - The same translation is appearing for multiple frames. If the user is continuing to do a gesture then it can possibly indicate they want the translation text displayed. 

 - The transaltion changes from the previous translation. A new gesture being read by the model thats different from the previous shows the user is doing a new motion to transalte. 

 - The user has a "Add Translation" button that will append the current translation being shown into the textbox when pressed. 
