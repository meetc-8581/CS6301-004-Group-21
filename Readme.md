## Installation

All the dependancies are listed in "requirements.txt" file.

## Dataset Download
Download the dataset from https://www.kaggle.com/datasets/jainishadesara/cnn-dailymail link.
Make a folder named "Data" in your working directory.
Put all the six downloaded data files in "Data" folder.

## Usage

Run the "distillation.py" to download and preprocess the data, to make the student model from teacher model, and finetune the student model to get the Rouge1, Rouge2, Rougel and RougeLSum scores as output in Result/metric.json file.


## Credits
We used some of the approach from the following repository:
https://github.com/huggingface/transformers/blob/e2c935f5615a3c15ee7439fa8a560edd5f13a457/examples/seq2seq

