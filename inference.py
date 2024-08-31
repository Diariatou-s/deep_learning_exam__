import os
import torch
import torchaudio
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer, FlaubertModel
from pydub import AudioSegment
import torch.nn as nn

# Define the device to use for computation (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomBert(nn.Module):
    """
    Custom BERT-based model for sentiment classification.
    """
    def __init__(self, name_or_model_path="flaubert/flaubert_base_uncased", num_classes=2):
        super(CustomBert, self).__init__()
        self.bert_pretrained = FlaubertModel.from_pretrained(name_or_model_path)
        self.classifier = nn.Linear(self.bert_pretrained.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # Forward pass through BERT model and classifier
        outputs = self.bert_pretrained(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.classifier(cls_output)
        return x
    
def classifier_fn(text, classification_model):
    """
    Classify the sentiment of the given text using the CustomBert model.
    """
    labels = {0: "negative", 1: "positive"}
    tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_uncased")
    
    # Tokenize the input text
    inputs = tokenizer(text, padding="max_length", max_length=250, truncation=True, return_tensors="pt")
    
    # Get model predictions
    output = classification_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    _, pred = output.max(1)
    sentiment = labels[pred.item()]
    
    return {"transcription": text, "sentiment": sentiment}

def convert_to_wav(m4a_path, wav_path):
    """
    Convert an audio file from m4a format to wav format.
    """
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    audio.export(wav_path, format="wav")

def process_audio_file(wav_file, model_sample_rate, processor, model):
    """
    Process a wav audio file to extract transcriptions using the ASR model.
    """
    # Load the audio file
    waveform, sample_rate = torchaudio.load(wav_file)

    # Resample the audio if necessary
    if sample_rate != model_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, model_sample_rate)
        waveform = resampler(waveform)

    # Prepare the input features for the model
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=model_sample_rate, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    # Generate transcriptions
    generated_ids = model.generate(input_features=input_features, max_new_tokens=225)
    generated_sentences = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_sentences

def main():
    # Load the sentiment analysis model
    classification_model = CustomBert()
    classification_model.load_state_dict(torch.load("model/model_flaubert.pth", map_location=torch.device('cpu')))
    
    # Load the ASR (Automatic Speech Recognition) model and processor
    model = AutoModelForSpeechSeq2Seq.from_pretrained("bofenghuang/whisper-small-cv11-french").to(device)
    processor = AutoProcessor.from_pretrained("bofenghuang/whisper-small-cv11-french", language="french", task="transcribe")
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="fr", task="transcribe")

    # Get the sampling rate used by the model
    model_sample_rate = processor.feature_extractor.sampling_rate

    # Define directories and supported file extensions
    input_directory = "files/audio_files/"
    wav_directory = "files/wav_files/"
    supported_extensions = (".m4a", ".wav", ".mp3", ".flac", ".ogg", ".aac")

    # Create the output directory if it does not exist
    os.makedirs(wav_directory, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(supported_extensions):
            input_path = os.path.join(input_directory, filename)
            wav_path = os.path.join(wav_directory, filename.rsplit(".", 1)[0] + ".wav")
            
            # Convert audio files to wav format
            convert_to_wav(input_path, wav_path)
            
            # Get transcription of the audio file
            transcription = process_audio_file(wav_path, model_sample_rate, processor, model)
            
            # Classify the sentiment of the transcription
            result = classifier_fn(transcription, classification_model)
            print(result)

if __name__ == "__main__":
    main()