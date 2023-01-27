# RGB2Spectral
Converting RGB photographs into full spectrum photography with Pix2PixHD and spectral reconstruction.<br>
This is not simply a color alteration of green to white. Pix2PixHD uses complex understandings of scenes to provide contextual conversion.
Unstructured local experimentation.

# Sample RGB:
<img src="1.jpg" alt="RGB" width="500"/>

# Sample Generated NIR:
<img src="1-IR1024.jpg" alt="NIR" width="500"/> <br>
Note the midground trees being lighter than the mountains in the background, contrary to the RGB brightness.
This is due to a contextual understanding of trees being brighter in NIR than mountains, as well as some cloud cover differences.<br>
Also note the tell-tale elimination of fog found in NIR photography due to IR fog penetration.

# Synthetic 590nm using composite of NIR and RGB based on spectral response of generic camera sensor:
<img src="1-590nm.JPG" alt="Composite 590nm" width="500"/>

# Example of in camera 590nm:
<img src="590nmSample.jpg" alt="Example 590nm" width="500"/>
https://kolarivision.com/wp-content/uploads/2022/05/590CWB-Medium-Small-Custom.jpg

# TODO:
  &emsp;Programatic spectral reconstruction and filtering<br />
  &emsp;Set up structured workflow<br />
  &emsp;Code cleanup<br />
  &emsp;Documentation<br />
