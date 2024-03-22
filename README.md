# Slash's Image Classification using TensorFlow

This project utilizes TensorFlow to perform image classification using a convolutional neural network (CNN). The CNN model is trained on a dataset of images belonging to four different categories: Beauty, Fashion, Home, and Nutrition. The trained model can then be used to predict the category of new input images.

## Approach

The image classification process follows these steps:
**Importing the needed libraries**: To import the the libraries i need i first installed tensorflow, opencv, and all matplotlib libraries.

1. **Data Preparation**: The dataset should be organized into four separate directories, each representing one category. The images should be in common formats such as JPEG, JPG, BMP, or PNG.

2. **Data Preprocessing**: The images are loaded using OpenCV and converted to the RGB color space. They are then normalized by dividing the pixel values by 255 to bring them within the range of 0 to 1. Additionally, the dataset is split into training, validation, and test sets.

3. **Model Architecture**: The CNN model consists of several convolutional layers followed by max-pooling layers to extract features from the input images. The extracted features are then flattened and passed through dense layers to perform classification. The final layer uses the softmax activation function to produce class probabilities.

4. **Model Training**: The model is compiled with the Adam optimizer and the categorical cross-entropy loss function. Training is performed for a specified number of epochs, and the model's performance is evaluated on the validation set after each epoch. The training history, including loss and accuracy, is recorded.

5. **Evaluation and Prediction**: After training, the model's performance can be evaluated on the test set using metrics such as precision, recall, and accuracy. Additionally, the model can make predictions on new images by resizing the images to the required input size and feeding them through the model. The predicted class is determined based on the class with the highest probability.

## Functionalities

The code provides the following functionalities:

- Loading and preprocessing the image dataset.
- Splitting the dataset into training, validation, and test sets.
- Creating and training a CNN model for image classification.
- Evaluating the model's performance on the test set.
- Making predictions on new images using the trained model.

## Practical Videos implementing the approach and functionalities

https://github.com/Geno212/Slash-s-Product-Image-Classifier/assets/127338940/2095a219-625a-4110-b53d-f8a9c14947b9



https://github.com/Geno212/Slash-s-Product-Image-Classifier/assets/127338940/ed6e746b-3bd2-44e0-8e68-b4a811e545ce

