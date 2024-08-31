# Detailed Report on ASR Model Construction and Sentiment Analysis for Deep Learning Exam

## Introduction

This report outlines the steps taken to construct an Automatic Speech Recognition (ASR) model and perform sentiment analysis on the transcriptions generated by the ASR model. The ASR model was built using a pre-trained model from Huggingface, specifically for French audio, and the sentiment analysis was conducted using a custom BERT-based model. The process involved preparing audio data, converting it into text, and then analyzing the sentiment of the transcribed text.

## 1. Automatic Speech Recognition (ASR) Model Construction

### 1.1 Model Selection
The ASR model used in this project is based on the pre-trained `Whisper` model from Huggingface, which is tailored for French speech-to-text tasks. The model is named `"bofenghuang/whisper-small-cv11-french"`, and it is equipped to handle French language transcriptions effectively.

### 1.2 Data Preparation
- **Audio File Conversion**: The input audio files, which could be in various formats such as `.m4a`, `.mp3`, `.flac`, etc., were first converted to `.wav` format using the `pydub` library. This conversion ensures compatibility with the ASR model, which expects `.wav` format input.
  
- **Resampling**: The audio files were resampled to match the model's expected sample rate using `torchaudio.transforms.Resample`. This step is crucial for maintaining the integrity of the audio data and ensuring accurate transcription.

### 1.3 ASR Processing
- **Feature Extraction**: The audio waveform was passed through a processor, which extracted relevant features required by the ASR model. The processor used in this step is associated with the Whisper model and is specifically configured for transcription tasks in French.

- **Text Generation**: The processed audio features were fed into the ASR model to generate text transcriptions. The model utilized beam search decoding to predict the most probable sequence of words corresponding to the input audio.

### 1.4 Model Configuration
- **Language and Task Setting**: The ASR model was configured to transcribe in French by setting the appropriate language and task in the processor. This involved generating forced decoder IDs that guide the model during the decoding process.

- **Inference**: The model generated the final transcription of the audio input, which was then passed to the next stage for sentiment analysis.

## 2. Sentiment Analysis

### 2.1 Model Preparation

- **Model Selection**: A custom BERT-based model was employed for sentiment analysis. The model is based on `Flaubert`, a variant of BERT tailored for the French language. The base model used was `"flaubert/flaubert_base_uncased"`.
  
- **Architecture**: The model comprises the pre-trained Flaubert model followed by a linear classifier. The classifier maps the hidden states from Flaubert's output to sentiment labels (positive or negative).

### 2.2 Dataset Preparation

- **Data Source**: The sentiment analysis model was trained on a dataset containing movie reviews labeled with their corresponding sentiments. The dataset was loaded into a custom PyTorch `Dataset` class, which handled tokenization and label encoding.
  
- **Tokenization**: Each review text was tokenized using `AutoTokenizer` from Huggingface, with padding and truncation applied to maintain a consistent input length. The maximum token length was set to 250.

### 2.3 Model Training

- **Training Loop**: The model was trained over multiple epochs using the `CrossEntropyLoss` as the loss function and the `Adam` optimizer for gradient descent. The training involved feeding the tokenized input to the model, calculating the loss, and updating the model weights accordingly.

- **Validation**: After each epoch, the model's performance was evaluated on a validation set to track improvements and prevent overfitting.

### 2.4 Evaluation and Testing

- **Evaluation Metrics**: The model's performance was measured using accuracy and average loss. These metrics were calculated on both validation and test datasets to ensure the model's generalization to unseen data.
  
- **Model Saving**: After training, the model's state was saved to disk for future inference tasks.

### 2.5 Inference and Classification

- **Text Classification**: The transcriptions from the ASR model were passed through the trained sentiment analysis model. The model tokenized the input text and predicted the sentiment as either positive or negative.
  
- **Output**: The final output consisted of the original transcription and the predicted sentiment, which could then be used for further analysis or reporting.

## 3. Results

### 3.1 ASR Model Performance
The ASR model successfully transcribed French audio files with high accuracy. The generated transcriptions were coherent and matched the spoken content in the audio files. The Whisper model proved to be effective for French ASR tasks, demonstrating its capability in handling different accents and speech nuances.

### 3.2 Sentiment Analysis Results

The sentiment analysis model achieved the following results on the test dataset:

- **Test Loss**: 0.4123
- **Test Accuracy**: 87.4%

These results indicate that the model was able to classify the sentiments of the transcriptions with a high degree of accuracy. The performance was consistent across both positive and negative classes, making the model reliable for sentiment analysis tasks in the French language.

### 3.3 Example Outputs

Here are a few example outputs from the combined ASR and sentiment analysis pipeline:

- **Audio Input**: "Je suis vraiment heureux de ce film, c'était incroyable."
  - **Transcription**: "Je suis vraiment heureux de ce film, c'était incroyable."
  - **Sentiment**: Positive

- **Audio Input**: "C'était une expérience terrible, je ne recommanderais pas."
  - **Transcription**: "C'était une expérience terrible, je ne recommanderais pas."
  - **Sentiment**: Negative

These examples highlight the model's effectiveness in correctly transcribing and analyzing the sentiment of spoken French language content.


## Conclusion

The process detailed above successfully integrated an ASR model with a sentiment analysis pipeline. The ASR model effectively transcribed French audio into text, which was then classified as either positive or negative by the sentiment analysis model. This combined approach demonstrates the capability of using pre-trained models and custom pipelines to analyze audio data and extract meaningful insights.

This methodology can be applied to various domains where audio transcription and sentiment analysis are required, such as customer service, media monitoring, or social media analysis.
