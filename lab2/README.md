# Fine-tune Whisper on your mothertongue (Italian)

Contributors:
<a href="https://github.com/matteocirca">Matteo Circa</a>, 
<a href="https://github.com/jiarre">Federico Giarrè</a>

## Lab 2

In this project we fine-tune <a href="https://huggingface.co/openai/whisper-small">OpenAI's Whisper model</a> for Italian automatic speech recognition (ASR). We do that by using the <a href="https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0">Common Voice dataset</a>, the Italian subset and by leveraging Hugging Face 🤗 Transformers.

## Implementation

The implementation consists in a feature engineering pipeline, a training pipeline and an inference program (Hugging Face Space). This is done to run the feature engineering on CPUs and the training pipeline on GPUs.
For the training pipeline we utilized the GPU given by a 2021 MacBook Pro with Apple M1 Max as a chip and 32GB of RAM.
 
1. We wrote a **feature engineering pipeline** that loads the dataset and takes 8h (~10%) of the Italian ASR dataset since it already yields good results and in a reasonable timing. We take both the train and test split to run some evaluation tests. Then, we load the Whisper Feature Extractor that pads the audio input to 30s and converts the inputs to Log-Mel spectrograms. We also load the Whisper Tokenizer which post-processes the model output to text format, which is our label.
We apply both the feature extractor and the tokenizer to our dataset to prepare for the traning. Finally we save the data locally, to be able in the next pipeline to load the data ready for the training, without going through all the previous steps.

2. We wrote a **training pipeline** that loads our dataset, consisting in 12360 rows for the training set and 5069 rows for the testing set, loads a pre-trained Whisper checkpoint <a href="https://huggingface.co/openai/whisper-small">Whisper small</a> and runs the training and evaluations to verify that we have correctly trained it to transcribe speech in Italian.
The evaluation uses the word error rate (WER) metric.

3. We wrote a first dummy Gradio application, our **inference program**, that:
- allows the user to speak into the microphone
- get the transcription of what is said

4. We then wrote our final Gradio application, our **inference program**, that:
- allows the user to speak into the microphone or upload an audio file and transcribe and translate what it's said or uploaded
- allows the user to search for a word in the transcription and get the timestamp of the word in the audio file

## Training parameters and possible improvements

The most relevant parameters we used are:
- `output_dir="./whisper-small-it"`: model predictions and checkpoints are written locally to be able to recover them and restart the training from the latest checkpoint.
- `evaluation_strategy="steps"`: the evaluation is performed (and logged) every `eval_steps=100` so that every 100 step of our training (1 steps has a batch size of 16) we compute the WER metric.
- `save_strategy="steps"`: our checkpoints are also saved every `save_steps=100` and by setting `save_total_limit=2` we limit the total amount of checkpoints, so that older checkpoints are deleted.

Possible improvements (**model-centric approach**):
- `num_train_epochs=1`: is the total number of training epochs to perform during training, we set it at 1 to be able to finish our training in reasonable timings.
We trained the model also with `num_train_epochs=2` getting a slightly better performance as we can see on the model training metrics: the training loss has improved in respect to the model trained with only one epoch (0.08137 vs. 0.1697).
- `learning_rate=1e-5`: we could try to fine-tune the value to choose the optimum one.
- Another way to improve our model performance is by selecting a larger pre-trained Whisper checkpoint for our training, like the `openai/whisper-medium` or `openai/whisper-large`. We couldn't test that for time constraints.

Possible improvements (**data-centric approach**): 
- Having more data is always a good approach so we could select a wider set from the Italian ASR dataset.

## Spaces on Hugging Face

### First dummy Gradio application - ASR App
https://matteocirca-asr-app.hf.space

### Final Gradio application - ASR App Pro
https://matteocirca-asr-app-pro.hf.space


## Models on Hugging Face

### whisper-small-it
https://huggingface.co/matteocirca/whisper-small-it

### whisper-small-it-2
https://huggingface.co/matteocirca/whisper-small-it-2
