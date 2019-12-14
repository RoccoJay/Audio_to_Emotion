# Classifying Audio to Emotion using RAVDESS

###### Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391.

This notebook includes code for reading in audio data, feature extraction, hyperparameter tuning with Optuna, and models including KNN, Logistic Regression, Decision Tree, Boosting, Bagging, Multilayer Perceptron, and Voting Classifiers. The Python library libROSA provided the main tools for processing and extracting features from the audio files utilized in this project.  

Beginning with extracting MFCCs, Chroma, and Mel spectrograms from the audio files modeling was done with readily available models from Sci-kit Learn and other Python packages. Hyperparameter tuning for these models was accomplished using the Optuna framework.

## Introduction 

Classifying audio to emotion is challenging because of its subjective nature. This task can be challenging for humans, let alone machines. Potential applications for classifying audio to emotion are numerous, including call centers, AI assistants, counseling, and veracity tests.  

There are numerous projects and articles available on this subject. Please see the references section at the bottom of this readme for useful and interesting articles and Jupyter notebooks on this or related topics.

Overview of this notebook's approach for classifying audio to emotion:
- Read WAV files in by using the libROSA package in Python.
- Extract features from the audio time series using functions from the libROSA package (MFCCs, Chroma, and Mel spectrograms).
- Construct a series of models from various readily-available Python packages.
- Tune hyperparameters for the models using the Optuna framework.
- Ensemble models using soft voting classifier to improve performance.

Audio is represented as waves where the x-axis is time and the  y-axis is amplitude.  These waves are  stored as a sum of sine waves using three values as in *A* sin(*B*t +*C*), where *A* controls the amplitude of the curve, *B* controls the period of the curve, and *C* controls the horizontal shift of the curve.  Samples are recorded at every timestep, and the number of samples per second is called the sampling rate, typically measured in hertz (Hz), which are defined as cycles per one second.  The standard sampling rate in libROSA is 22,050 Hz because that is the upper bound of human hearing.

### Data Description

The RAVDESS dataset consists of speech and song files classified by 247 untrained Americans to eight different emotions at two intensity levels: Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprise, along with a baseline of Neutral for each actor. A breakdown of the emotion classes in the dataset is provided in the following table:

| Emotion | Speech Count | Song Count | Total Count 
| ------- | :----------: | :--------: | ----------: 
| Neutral | 96 | 92 | 188
| Calm | 192 | 184 | 376
| Happy | 192 | 184 | 376
| Sad | 192 | 184 | 376
| Angry | 192 | 184 | 376
| Fearful | 192 | 184 | 376
| Disgust | 192 | 0 | 192
| Surprised | 192 | 0 | 192
| Total | 1440 | 1012 | 2452

The dataset is composed of 24 professional actors, 12 male and 12 female making it gender balanced. 

The audio files were created in a controlled environment and using identical statements spoken in an American accent. There are two distinct types of files:
- Speech file (Audio_Speech_Actors_01-24.zip, 215 MB) contains 1440 files: 60 trials per actor x 24 actors = 1440. 
- Song file (Audio_Song_Actors_01-24.zip, 198 MB) contains 1012 files: 44 trials per actor x 23 actors = 1012.  

The files are in the WAV raw audio file format and all have a 16 bit Bitrate and a 48 kHz sample rate. The files are all uncompressed, lossless audio, and have not lost any information/data or been modified from the original recording. 

## Feature Extraction

As mentioned before, the audio files were processed using the libROSA python package. This package was originally created for music and audio analysis, making it a good selection. After importing libROSA, the WAV files are read in one at a time. An audio time series in the form of a 1-dimensional array for mono or 2-dimensional array for stereo, along with time sampling rate (also defines the length of the array), where the elements within each of the  arrays represent the amplitude of the sound waves is returned by libROSA’s “load” function.

Some helpful definitions for understanding the features used:

- **Mel scale** — deals with human perception of frequency, it is a scale of pitches judged by listeners to be equal distance from each other
- **Pitch** — how high or low a sound is. It depends on frequency, higher pitch is high frequency
- **Frequency** — speed of vibration of sound, measures wave cycles per second
- **Chroma** — Representation for audio where spectrum is projected onto 12 bins representing the 12 distinct semitones (or chroma). Computed by summing the log frequency magnitude spectrum across octaves.
- **Fourier Transforms** — used to convert from time domain to frequency domain
  - *time domain*: shows how signal changes over time
  - *frequency domain*: shows how much of the signal lies within each given frequency band over a range of frequencies

Using the signal extracted from the raw audio file and several of libROSA’s audio processing functions, MFCCs, Chroma, and Mel spectrograms were extracted using a function that receives a file name (path), loads the audio file, then utilizes several libROSA functions to extract features that are then aggregated and returned in the form of a numpy array.

### Summary of Features

- **MFCC** - Mel Frequency Cepstral Coefficients: 
Voice is dependent on the shape of vocal tract including tongue, teeth, etc.
Representation of short-time power spectrum of sound, essentially a representation of the vocal tract

- **STFT** - returns complex-valued matrix D of short-time Fourier Transform Coefficients:
Using abs(D[f,t]) returns magnitude of frequency bin f at frame t (Used as an input for Chroma_STFT)

- **Chroma_STFT** - (12 pitch classes) using an energy (magnitude) spectrum (obtained by taking the absolute value of the matrix returned by libROSA’s STFT function) instead of power spectrum returns normalized energy for each chroma bin at each frame

- **Mel Spectrogram** - magnitude spectrogram computed then mapped onto mel scale—x-axis is time, y-axis is frequency

## Modeling

After all of the files were individually processed through feature extraction, the dataset was split into an 80% train set and 20% test set. This split size can be adjusted in the data loading function.

A Breakdown of the Models:
- Simple models: K-Nearest Neighbors, Logistic Regression, Decision Tree
- Ensemble models: Bagging (Random Forest), Boosting (XG Boost, LightGBM)
- Multilayer Perceptron Classifier
- Soft Voting Classifier ensembles

The hyperparameters for each of the models above were tuned with the Optuna framework using the mean accuracy of 5-fold cross-validation on the train set as the metric to optimize. This particular framework was chosen due to its flexibility, as it allows for distributions of numerical values or lists of categories to be suggested for each of the hyperparameters. Its pruning of unpromising trials makes it faster than a traditional grid search.

## Results
The results and parameters of the top performing models are provided below, as well as a summary of metrics obtained by other models. Note that results will vary slightly with each run of the associated Jupyter notebooks, unless seeds are set.  Overfitting was an issue with the majority of the models with some models overfitting to a greater or lesser degree than others.  This may have been caused in part by the relatively small size of the dataset. 

### XG Boost

The following parameters were obtained for the XG Boost model using the Optuna framework, and yielded a test set accuracy of 0.73.  

| Parameter | Value
| --- | ---
| booster | gbtree
| lambda | 7.20165E-08
| alpha | 2.24951E-05
| max_depth | 7
| eta | 9.30792E-06
| gamma | 1.79487E-05
| grow_policy | lossguide

This model has more trouble classifying fearful female, sad female, and sad male than some of the other classes.  

### Multilayer Perceptron

The following parameters were obtained for the MLP model using the Optuna framework, and yielded a test set accuracy of 0.83. 

| Parameter | Value
| --- | ---
| activation | relu
| solver | lbfgs
| hidden_layer_size | 1283
| alpha | 0.3849486
| batch_size | 163
| learning_rate | constant
| max_iter | 1000

The model showed considerable improvement in classifying fearful female and sad male, but has even more trouble classifying sad female than the XG Boost model.  

### Soft Voting Classifier Ensemble (MLP and XGB)

Thinking that the performance of the MLP model could offset the poor performance of the XG Boost model in the fearful female and sad male classes and vice versa with the performance on the sad female class, a soft voting classifier was used to average the probabilities produced by each of the models.  This voting classifier outperformed each of the component models in both the 5-fold CV accuracy over the train set and the test set accuracy obtaining 0.84.  Although the voting classifier model performs better in fearful female, sad female, and sad male, it still struggled with sad female as the component models did.

Many other models were trained, tuned, and tested besides those discussed previously.  Below is a summary of the statistics associated with these other models.

| Model | Train | CV 5-fold on Train | Test
| --- | --- | --- | ---
| Decision Tree | 0.74 | 0.431 | 0.46 
| Random Forest | 1.00 | 0.639 | 0.67
| LGBM Classifier | 1.00 | 0.658 | 0.71
| XGB Classifier | 1.00 | 0.656 | 0.73
| MLP Classifier | 1.00 | 0.767 | 0.83
| KNN | 1.00 | 0.524 | 0.55
| Logistic Regression | 0.82 | 0.643 | 0.68
| Voting Classifier 1: mlp, lgb | 1.00 | 0.765 | 0.82
| Voting Classifier 2: kn, xgb, mlp | 1.00 | 0.767 | 0.83
| Voting Classifier 3: xgb, mlp, rf, lr | 1.00 | 0.751 | 0.82
| Voting Classifier 4: mlp, xgb | 1.00 | 0.769 | 0.84

## Conclusion

The use of three features (MFCC’s, Mel Spectrograms and chroma STFT) gave impressive accuracy in most of the models, reiterating the importance of feature selection.  As with many data science projects, different features could be used and/or engineered.  Tonnetz was originally used in modeling, however it lead to decreased performance and was removed. Some other possible features to explore concerning audio would be MFCC Filterbanks or features extracted using the perceptual linear predictive (PLP) technique.  These features could affect the performance of models in the emotion classification task.  

## Future Work
An alternate approach that could be explored for this problem is splitting the classifying task into two distinct problems.  A separate model could be used to classify gender and then separate models for each gender to classify emotion could be utilized.  This could possibly lead to a performance improvement by segregating the task of emotion classification by gender.

It would be interesting to see how a human classifying the audio would measure up to these models, however, finding someone willing to listen to more than 2,400 audio clips may be a challenge in of itself because a person can only listen to “the children are talking by the door” or “the dogs are sitting by the door” so many times.

## References
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
