# Classifying Audio to Emotion using RAVDESS

This notebook includes code for reading in audio data, feature extraction, hyperparameter tuning with Optuna, and models including KNN, Logistic Regression, Decision Tree, Boosting, Bagging, Multilayer Perceptron, and Voting Classifiers.The Python library libROSA provided the main tools for processing and extracting features from the audio files utilized in this project.  

Beginning with extracting MFCCs, Chroma, and Mel spectrograms from the audio files modeling was done with readily available models from Sci-kit Learn and other Python packages.  Hyperparameter tuning for these models was accomplished using the Optuna framework.

## Introduction & Background 

Classifying audio to emotion is challenging because of its subjective nature.  This task can be challenging for humans, let alone machines.  Potential applications for classifying audio to emotion are numerous, including call centers, AI assistants, counseling, and veracity tests.  

There are numerous projects and articles available on this subject.  Please see the references section at the bottom of this readme for useful and interesting articles and Jupyter notebooks on this or related topics.

Overview of this notebook's approach for classifying audio to emotion:
-Read WAV files in by using the libROSA package in Python.
-Extract features from the audio time series using functions from the libROSA package (MFCCs, Chroma, and Mel spectrograms).
-Construct a series of models from various readily-available Python packages.
-Tune hyperparameters for the models using the Optuna framework.
-Ensemble models using soft voting classifier to improve performance.

Audio is represented as waves where the x-axis is time and the  y-axis is amplitude.  These waves are  stored as a sum of sine waves using three values as in A sin(\omegat +\phi), where A controls the amplitude of the curve, \omega controls the period of the curve, and \phi controls the horizontal shift of the curve.  Samples are recorded at every timestep, and the number of samples per second is called the sampling rate, typically measured in hertz (Hz), which are defined as cycles per one second.  The standard sampling rate in libROSA is 22,050 Hz because that is the upper bound of human hearing.

### Data Description
The RAVDESS dataset consists of speech and song files classified by 247 untrained Americans to eight different emotions at two intensity levels: Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprise, along with a baseline of Neutral for each actor. A breakdown of the emotion classes in the dataset is provided in the following table:
| Emotion | Speech Count | Song Count | Total Count |
|---------|--------------|------------|-------------|
Neutral
96
92
188
Calm
192
184
376
Happy
192
184
376
Sad
192
184
376
Angry
192
184
376
Fearful
192
184
376
Disgust
192
0
192
Surprised
192
0
192
Total
1440
1012
2452

The dataset is gender balanced being composed of 24 professional actors, 12 male and 12 female. 

The audio files were created in a controlled environment and each consists of identical statements spoken in an American accent. Additionally, there are two distinct types of files:
Speech file (Audio_Speech_Actors_01-24.zip, 215 MB) contains 1440 files: 60 trials per actor x 24 actors = 1440. 
Song file (Audio_Song_Actors_01-24.zip, 198 MB) contains 1012 files: 44 trials per actor x 23 actors = 1012.
The files are in the WAV raw audio file format and all have a 16 bit Bitrate and a 48 kHz sample rate. The files are all uncompressed, lossless audio, meaning that the audio files in the dataset have not lost any information/data or been modified from the original recording.   
As mentioned before, to process/manipulate these files we used the libROSA python package. This package was originally created for music and audio analysis, making it the perfect selection for dealing with our dataset.
After importing libROSA, we read in one WAV file at a time.  An audio time series in the form of a 1-dimensional array for mono or 2-dimensional array for stereo, along with time sampling rate (which defines the length of the array), where the elements within each of the  arrays represent the amplitude of the sound waves is returned by libROSA’s “load” function.
Data Pre-Processing & Exploration
Before going into pre-processing and data exploration we will explain some of the concepts that allowed us to select our features.
Mel scale — deals with human perception of frequency, it is a scale of pitches judged by listeners to be equal distance from each other
Pitch — how high or low a sound is. It depends on frequency, higher pitch is high frequency
Frequency — speed of vibration of sound, measures wave cycles per second
Chroma — Representation for audio where spectrum is projected onto 12 bins representing the 12 distinct semitones (or chroma). Computed by summing the log frequency magnitude spectrum across octaves.
Fourier Transforms — used to convert from time domain to frequency domain
o   time domain: shows how signal changes over time
o frequency domain: shows how much of the signal lies within each given frequency band over a range of frequencies

Fourier Transform going from time domain to frequency domain
Using the signal extracted from the raw audio file and several of libROSA’s audio processing functions, MFCCs, Chroma, and Mel spectrograms were extracted using the following function: 

The function receives a file name (path) and loads the audio file using the libROSA library. Several libROSA functions are utilized to extract features that are then aggregated and returned in the form of a numpy array.
The spectrograms used in our VGG 16 model discussed later were made using the following class coded here.

Summary of the Features
RAW AUDIO - Image output of the audiofile read in by libROSA:

MFCC - Mel Frequency Cepstral Coefficients: 
Voice is dependent on the shape of vocal tract including tongue, teeth, etc.
Representation of short-time power spectrum of sound, essentially a representation of the vocal tract

STFT - returns complex-valued matrix D of short-time Fourier Transform Coefficients:
Using abs(D[f,t]) returns magnitude of frequency bin f at frame t


CHROMA_STFT - (12 pitch classes) using an energy (magnitude) spectrum (obtained by taking the absolute value of the matrix returned by libROSA’s STFT function) instead of power spectrum returns normalized energy for each chroma bin at each frame


MEL SPECTROGRAM - magnitude spectrogram computed then mapped onto mel scale—x-axis is time, y-axis is frequency


Modeling
After all of the files were individually processed through feature extraction, the dataset was split into train and test in an 80-20 split. 
The modeling process was divided into two main parts: “traditional” machine learning models and deep neural networks. Simpler models were to be used as a baseline for the convolutional neural network and recurrent neural network. 
Traditional Machine Learning Models:
Simple models: K-Nearest Neighbors, Logistic Regression, Decision Tree
Ensemble models: Bagging (Random Forest), Boosting (XG Boost, LightGBM)
Multilayer Perceptron Classifier
Soft Voting Classifier ensembles
The hyperparameters for each of the models above were tuned with the Optuna framework using the mean accuracy of 3 to 5 fold cross-validation on the train set as the metric to optimize. This particular framework was chosen due to its flexibility, as it allows for distributions of numerical values or lists of categories to be suggested for each of the hyperparameters, and because it prunes the unpromising trials.
Deep Learning:
Two different approaches in modeling with deep learning networks were pursued:
Design a convolutional neural network and train it on a combination of MFCC, Mel Scale, and Chroma features
Take a more robust, widely tested convolutional architecture and train it on Mel spectrograms
Approach 1:
For this approach, we began with MFCCs, Mel Scale, and Chroma features (180 features in total). Following the same approach as this research paper, we reduced the number of features while attempting to keep as much information as possible using the dimensional reduction technique called Principal Component Analysis (PCA). Finally, the new feature set had 66 features capturing 95% of the variance in the original set.
We experimented with multiple combinations of the number and size for convolution layers and fully-connected layers, optimizers, batch sizes, and epochs to get the best performance. Finally, a neural network was designed with:
8 convolution layers and 3 fully connected layers
Relu as the activation function
Batch normalization and dropouts at different stages
SGD optimizer with momentum=0.9 and learning_rate=0.01
Categorical Cross-entropy as the loss function
Batch size as 16 
Softmax function for final 16-class classification
 The architecture can be seen below:

The convolution layers work to extract high-level complex features from the input data while the fully-connected layers are used to learn non-linear combinations of these features that will be used for classification.
Some key points about the input and output of the network:
It takes in the input of an (n, 66, 1) shaped array where n is the number of audio files (1961 in our case for the train set)
The Output of the network is an (n, 16) array which gives us the probabilities associated with each emotion for audio files
The argmax() function was used to find the emotion with the maximum probability
Approach 2:
In an alternative approach, we made use of a widely used network and trained it on our data.  We tried different, well-known, architectures of CNNs such as VGG 16, VGG 19, ResNet 50, and InceptionNet and, finally, settled on VGG 16 as it was fairly robust and the most feasible option.
In this network, we used the bottleneck layers of VGG 16 to extract features from Mel spectrograms and added the same set of fully-connected layers, optimizer, batch size, and loss function used in Approach 1. 
Mel spectrograms were converted to 224*224 pixel images. These were then used as input to VGG 16 network. The output is of the same format as in the previous approach.
Some interesting findings:
The models from both approaches were found to be overfitting during training. To avoid this we tried more aggressive dropouts, L1 and L2 regularization at various convolution and fully-connected layers for Approach 1 and only on the fully-connected layer for Approach 2. However, this resulted in a significant decrease in both training and validation accuracies even with an extremely small value of lambda.
Batch size also had an impact on model accuracy. A decrease in batch size led to an increase in accuracy, however, this increased the training time non-linearly. So after multiple iterations, we fixed the batch size to 16. 
Final Approach:
In our final approach, we decided to create an ensemble of the neural nets developed in Approach 1 and Approach 2. To do this, we used the soft voting technique to combine the resultant posterior probabilities from both models. We found that giving a weight of three to posterior probabilities from Approach 1 and weight of two to posterior probabilities from Approach 2 resulted in better overall accuracy.
Results
The results and parameters of the top performing models are provided below, as well as a summary of metrics obtained by other models. Note that results will vary slightly with each run of the associated Jupyter notebooks, unless seeds are set.  Overfitting was an issue with the majority of our models with some models overfitting to a greater or lesser degree than others.  We believe this may have been caused in part by the relatively small size of the dataset.  Below is some of the code used to train and test the traditional machine learning models.



To compare the different models we had to choose a common metric, in this case we decided on using the overall accuracy of the model since we are weighting all classes the same. The accuracy is a simple metric to compute across all the models as it can be done from the confusion matrix by simply adding the values in the diagonal over the total number of points. 
Traditional Machine Learning

XG Boost

	The following parameters were obtained for the XG Boost model using the Optuna framework, and yielded a test set accuracy of 0.73.  






Parameter
Value
booster
gbtree
lambda
7.20165E-08
alpha
2.24951E-05
max_depth
7
eta
9.30792E-06
gamma
1.79487E-05
grow_policy
lossguide

Below is the confusion matrix produced by this model.  The confusion matrix shows that this model has more trouble classifying fearful female, sad female, and sad male than some of the other classes.  

MLP
The following parameters were obtained for the MLP model using the Optuna framework, and yielded a test set accuracy of 0.83. 





Parameter
Value
activation
relu
solver
lbfgs
hidden_layer_size
1283
alpha
0.3849486
batch_size
163
learning_rate
constant
max_iter
1000

Below is the confusion matrix produced by the MLP model.  The model shows considerable improvement in classifying fearful female and sad male, but has even more trouble classifying sad female than the XG Boost model.  

Soft Voting Classifier Ensemble (MLP and XGB)
Thinking that the performance of the MLP model could offset the poor performance of the XG Boost model in the fearful female and sad male classes and vice versa with the performance on the sad female class, a soft voting classifier was used to average the probabilities produced by each of the models.  This voting classifier outperformed each of the component models in both the 5-fold CV accuracy over the train set and the test set accuracy obtaining 0.84.  The confusion matrix shows that although the voting classifier model performs better in fearful female, sad female, and sad male, it still struggled with sad female as the component models did.

Many more traditional machine learning models were trained, tuned, and tested other than those discussed previously.  Below is a summary of the statistics associated with these other models.
Model
Train
CV 5-fold on Train
Test
Decision Tree
0.74
0.431
0.46
Random Forest
1.00
0.639
0.67
LGBM Classifier
1.00
0.658
0.71
XGB Classifier
1.00
0.656
0.73
MLP Classifier
1.00
0.767
0.83
KNN
1.00
0.524
0.55
Logistic Regression
0.82
0.643
0.68
V1: mlp, lgb
1.00
0.765
0.82
V2: kn, xgb, mlp
1.00
0.767
0.83
V3: xgb, mlp, rf, lr
1.00
0.751
0.82
V4: mlp, xgb
1.00
0.769
0.84


Deep Networks
Convolutional Neural Network
Below is the confusion matrix produced by the Convolutional Neural Network.  In addition to having issues classifying sad female and fearful female as is the case with most of the models, it shows that this model also struggles to classify angry male, fearful male, and happy male.	

VGG 16
Below is the confusion matrix produced by the VGG 16.  It shows comparable performance to the CNN in classifying sad female and fearful female while being less accurate with fearful male and sad male; however the model shows marked improvement in classifying angry male and happy male, as well as small improvements in several other classes.

Soft Voting Classifier Ensemble (CNN and VGG 16)
The confusion matrix for the Soft Voting Ensemble of the CNN and VGG 16 models shows improvement over its component models in classifying fearful female, sad female, fearful male and sad male, while sacrificing some accuracy with classifying happy male.  Overall the ensemble boasts a better accuracy than either of its component models.

	Several other deep learning models were trained and tested throughout this project. The results are summarized in the table below.  
Model
Train Accuracy
Test Accuracy
CNN
1
0.68
VGG 16
1
0.73
RNN
0.19
0.18
MobileNet
0.99
0.56
VGG 19 PT
0.99
0.59
VGG 19
1
0.46
V: VGG 16, CNN
1
0.78

As reflected in the results tables for both traditional machine learning and deep learning, the highest accuracies for both approaches were achieved by using soft voting classifiers. It is interesting to note that some of the simpler models like logistic regression performed with comparable accuracy to the CNN. This could be due to the relatively small size of the dataset and/or the number of epochs used in training the CNN.
	On this particular dataset a simple neural net, such as a multilayer perceptron, which we treated as a traditional machine learning approach, was the top performer on its own.
Some other interesting results are:
The deep networks trained on spectrograms performed poorly compared to the ensembles trained on a collection of features.
Adding song data to increase the size of the dataset improved the model performance across the board even though it caused a slight imbalance of the classes.
Conclusion
This project started with the desire to understand how to implement and use deep networks, however throughout the course of the project the importance of feature engineering became predominant. 
It is abundantly clear from observing that the RNN trained using aggregated features performed worse than simple logistic regression that complex models are not always the best performing. Even when using the same features on an MLP and a CNN, the MLP outperformed the more complex model. As mentioned before, this could be because the size of the dataset was insufficient to properly train a deeper network.
Not only does the MLP perform better, but it also takes less time and effort to train once the features have been selected and/or created. This by itself can be very significant when deploying models in a business environment.
The use of three features (MFCC’s, MSF’s and chroma STFT) gave impressive accuracy in both simple and deep learning models, reiterating the importance of feature selection and understanding the data in order to select the proper preprocessing methods.
Future Work
	An alternate approach that could be explored for this problem is splitting the classifying task into two distinct problems.  A separate model could be used to classify gender and then separate models for each gender to classify emotion could be utilized.  This could possibly lead to a performance improvement by segregating the task of emotion classification by gender.

	As with many data science projects, different features could be used and/or engineered.  Some possible features to explore concerning speech would be MFCC Filterbanks or features extracted using the perceptual linear predictive (PLP) technique.  These features could affect the performance of models in the emotion classification task.

	It would be interesting to see how a human classifying the audio would measure up to our models, however, finding someone willing to listen to more than 2,400 audio clips may be a challenge in of itself because a person can only listen to “the children are talking by the door” or “the dogs are sitting by the door” so many times.
References
https://zenodo.org/record/1188976#.XeqDKej0mMo
http://conference.scipy.org/proceedings/scipy2015/pdfs/brian_mcfee.pdf
https://towardsdatascience.com/ok-google-how-to-do-speech-recognition-f77b5d7cbe0b
http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
https://en.wikipedia.org/wiki/Frequency_domain 
http://www.nyu.edu/classes/bello/MIR_files/tonality.pdf
https://github.com/marcogdepinto/Emotion-Classification-Ravdess/blob/master/EmotionsRecognition.ipynb
https://towardsdatascience.com/speech-emotion-recognition-with-convolution-neural-network-1e6bb7130ce3
https://data-flair.training/blogs/python-mini-project-speech-emotion-recognition/
https://labrosa.ee.columbia.edu/matlab/chroma-ansyn/
https://librosa.github.io/librosa/index.html
https://www.researchgate.net/publication/283864379_Proposed_combination_of_PCA_and_MFCC_feature_extraction_in_speech_recognition_system
