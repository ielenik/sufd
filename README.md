# surveillance face detector
This is a face detection filter designed specifically for surveillance cameras. 

Its advantages include 
-- very fast processing (capable of processing a 256px image in 0.4ms on an NVIDIA 1080, which equates to around 2500 fps)
-- providing a rotated square bounding box for further face recognition
-- offers additional face attributes such as the probability of spoofing and mask wearing

However, the filter does not work with large faces (although they can be found by reducing the image size). It also has difficulty detecting faces that are very close to each other, and the spoof filter is known to be unstable. 


I will not publish the dataset used to train the network becouse it is a mix of in-house and external images, but a training script is available for users to train their own network on their own data. 

The network itself is relatively simple, consisting of a few convolutional layers with several pooling layers that generate a feature map to detect faces and regress their size, angle, and attributes. 
During inference, the filter uses max pooling instead of non-max suppression to identify pixels with high probabilities of containing faces. 

An example of using the filter can be found in the testvideo.py file.

