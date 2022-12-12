# Classifying Art

### Introduction

Art can often be a fickle topic — its intrinsic value, restricted to canvas and oil-based paint, is rather opaque. Ranging from Modernist interpretations to even heralded classics of the Renaissance, art has always been difficult to decipher. Its universal critique, across ages, is evidence of this fact. Especially to the untrained eye, potentially those of a Computer Science enthusiast, it is rather difficult to discern genres and movements spanning decades and even centuries, even despite underlying appreciation and admiration.

Imagine a scenario: it’s another post-graduation alumni meet up and this time it’s hosted at the local art museum. Your last encounter with art was “Art History: Impressionism in the early 1900s” with Professor Schumer, so you’re a tad rusty. If only there was a simple way to aid you with navigating through all of the paintings.

Utilizing deep learning technologies, we can accomplish just this. Without the need for years of in-depth analysis and research, we can expand knowledge to a wider range of individuals, increasing accessibility to the arts that so people lack.

### Methodology

Our approach to solving this problem was using Convolutional Neural Networks (CNN). As CNN is a well-known architecture used for image recognition and processing pixel data, we thought it would be a good approach for this task. We had also researched other Deep Learning projects which involved paintings, and seen a common trend in the use of CNN, and more specifically Residual Neural Networks (ResNet). 

##### Data

The dataset we're using is from Kaggle, scraped from artchallenge.ru at the end of February 2019 by the user ICARO. The dataset itself contains 1000+ paintings from 50 artists. This dataset contains three files: artists.csv, images.zip, and resized.zip.

Artists.csv allowed us to extract the data we wanted, and create labels with respect to each painting. The original images folder was a folder structure with all the images split up by their artist and the resized folder was the same images not in the folder structure and made smaller in order to process easier. 

Images were given in a folder structure of artist to paintings. Intensive pre-processing was required to rearrange the folder structure to correspond to each paintings’ genre, not their artist. Due to computational limitation and prediction accuracy, we only chose genres with 200+ paintings: Impressionism, Post-Impressionism, Cubism, Northern Renaissance, High Renaissance, Baroque, Primitive, Romanticism, Surrealism. Additionally, at random, only 25% of Impressionism and Post-Impressionism were selected due to a severely unbalanced dataset.


##### Preprocessing
Our dataset consisted of artists, the artist's respective painting genre, a collection of paintings by the artists, and other information about the artist. We only needed the painting and the respective genre so we began by finding all artists that painted under only one genre. This was important as paintings that could fall under multiple categories should be removed as it would hinder the learning process. There were a total of 20 painting genres, and each painting had a different number of paintings. We also selected only 25% of Impressionism and Post-Impressionism paintings to reduce the load. 

The images were also provided in a resized manner which was necessary as the sizes of the original images were too large. 

##### Augmentation

We had a 80/20 split on training/validation data and we augmented the training data. Our augmentation included rescaling, shear (shear_range=5), zoom (zoom_range=0.2), horizontal, and vertical flips on all of the images. 

##### Sequential Model

We ran the ResNet50’s CNN which includes 50 dense layers that are trained on the ImageNet database, a database with more than 14 million images. This makes ResNet50’s weights already pre-trained on millions of images. Still, however, we added two sets of dense layers for better classification on our data set. Both dense layers were followed by Batch Normilzation and an LeakyReLU. 

The training was run on 10 epochs (2 hours) and we used an Adam loss function with a learning rate of 0.0001. The metric being measured was categorical accuracy. 

### Results 
Prior to testing, we laid out the following target goals for our categorical accuracy:
  At least 60% categorical accuracy for training set
  At least 40% categorical accuracy for testing set
We were able to overachieve on both of these:
  Training Evaluation: 99.64% (categorical accuracy)
  Testing Evaluation: 58.14% (categorical accuracy)

Our confusion matrix also showed certain standout accuracies such as 93% on Northern Renaissance and 83% on Cubism. There were many other patterns we notice such as Impressionism being predicted Post-Impressionsim 80% of the time


### Challenges

One of the main challenges as with many machine learning models, was improving accuracy. As we started with a very low accuracy we worked on tweaking preprocessing and seeing error in our data to help mitigate the effects of this. We saw a huge improvement after reducing the number of Dense layers to keep the model from overfitting. We also went ahead and changed our input by only using a more equal amount of images per genre. We removed outliers on the higher and lower ends and also got rid of our previous “weights” per genre and this helped our model by reducing overfitting and removing labels that didn’t have enough corresponding testing data. 

##### Reflection
Our reach goal was to achieve atleast 40% categorical accuracy on the evaluation. The reason we made this a reach goal is that we believed that classifying paintings to a specific genre isn’t simple even for a trained eye. Some factors such as contrast, temperature, colors, and saturation might be well-depicted through the pixels. Other such as brush strokes and precision (of the intended painting subject) aren’t quantified as easily and therefore cannot be accounted for as strongly. 
 
Another factor we played with was the number of output classes, in our case the number of painting genres. Starting with 20 genres and many other parameters different we were able to hit a categorical accuracy of ~63% compared to ~99% with 8 genres. This was primarily due to the inconsistent number of data points per label, and by increasing the minimum number of data points and scaling down outliers it became a far more even dataset that wasn’t overfitting. Almost certainly, more labels would bring down the accuracy, but if we were t revert back to our model with 20 labels, but an even number of each painting we felt like we could have a far more accurate model, so in this case, the lack of data limited us slightly. 

The Confusion matrix provided some interesting insights into the similarity between genres. One thing that was very interesting was how only 18% of Impressionism was predicted correctly and it was labeled as Post-Impressionism 80% of the time. This could be due to the fact that Impressionism could almost be classified as a manner in which one paints, as it rather than a style of painting. It would be interesting to see how things would change if Impressionism and Post-Impressionsim were labelled as the same category. 

Another notable thing found in the Confusion matrix was the most “unique” styles of painting. Northern Renaissance and Cubism both were very successful in predicting the styles with 93% and 83% respectively, but both are widely considered unique styles of painting. 
