# The prototype :

![WhatsApp Image 2024-05-30 at 23 30 02_4a0d77e8](https://github.com/valid999/Autonomous_system/assets/95305177/07ba266a-dae5-45d8-b397-d4e1ede445cb)

![WhatsApp Image 2024-05-27 at 19 52 36_a8551c9d](https://github.com/valid999/Autonomous_system/assets/95305177/dd9c3676-2360-4928-a1f5-1641e23a60a8)


# Final_Project Self driving car system with neural network

The article provides and depth overview of the system vision , development and the applied deep learning algorithm inside the self-driving car system , while there is a lot of challenges but in this project I will try my best to prove if we negate the some expensive sensor of the car we can get a huge result from the system vision , also I placed several sensor to show we can take the experiment with sensor and then without sensor the idea will start by showing how if we placed 6+ camera in the car can observe the environment and take 33 images per second can give us a huge result. Second will see the different understandable from the normal system and the vision system and how we can avoid the cost of the cars if we integrated this model in different area can- perform good .
Overall , this paper provides readers with a thorough understanding of current state-of- art technology related to self driving car ,fabricating, and deploying the model inside the controller systems specifically designed for  self driving car system . it is insightful to catch the huge development of the deep learning and  apply it in the real world , These machines offer over traditional methods while acknowledging challenges faced by researchers working in the field  is how to make the deployment so efficient  over traditional methods.

# 1. Data collection 
Data Collection: The system captures real-time data from cameras mounted on the vehicle maybe the number of these cameras will be 8 camera. These cameras provide a continuous stream of images representing the vehicle's environment.
Data Preprocessing: The captured images are preprocessed to enhance features and reduce noise. Techniques such as normalization, resizing, and augmentation (flipping, rotation, etc.) are applied to improve the robustness of the CNN model , the main technique we are using right now is a technique related to the  PyTorch library will done everything behind the seen in terms of the resizing and shaping of the data we need to dataset with annotation.
Model Architecture : A CNN model is designed with multiple convolutional layers to extract spatial features from images. This model includes layers for convolution, pooling, and fully connected operations, which enable the extraction of hierarchical features from raw pixel data , there is also a flaaten layer that will give us to take the data from the normal layers into the fully connected layers we can say the flatten layer stack the data from the CNN into fully connected layer and I will mention it by details.
Training : The CNN model is trained on a labeled dataset, where each image is associated with a corresponding s teering angle or driving command. The training process involves backpropagation and optimization algorithms (e.g., Adam) to minimize the error between the predicted and actual driving commands , the training will quite difficult because of the GPU  and also base don the number of the dataset , The backpropagation stage is done by the PyTorch even the setting the gradient to zero.
Testing and Validation : The trained model is validated using a separate dataset to evaluate its performance in real-world scenarios. Metrics such as accuracy, precision, recall, and mean squared error are used to assess the model's effectiveness.
Deployment : The validated model is deployed in a self-driving car system ,  where it processes live camera feeds to make real-time driving decisions .
Some  benefits:
Enhanced Safety : Self--driving cars can reduce human errors , which is a leading cause of traffic accidents . CNNs enable these vehicles to accurately interpret and respond to complex visual stimuli , enhancing overall road safety .
Increased Efficiency : Autonomous vehicles can optimize driving patterns and reduce traffic congestion by communicating with each other and adjusting their routes in real-time.
Accessibility : Self--driving technologys can provide mobility solutions for individuals who are unable to drive, such as the elderly or disabled ,thereby increasing their independence and quality of life .
Environmental Impact : Autonomous vehicles can contribute to reduced emissions by optimizing driving efficiency and supporting the integration of electric vehicles.




# 2. Object detection using the neural network(CNN).

https://youtu.be/1xrBWL5Kkog

