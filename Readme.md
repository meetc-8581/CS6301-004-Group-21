## Installation

All the dependancies are listed in "requirements.txt" file.

## Dataset Download
Download the dataset from https://www.kaggle.com/datasets/jainishadesara/cnn-dailymail link.
Make a folder named "Data" in your working directory.
Put all the six downloaded data files in "Data" folder.

## Usage

Run the "distillation.py" to download and preprocess the data, to make the student model from teacher model, and finetune the student model to get the Rouge1, Rouge2, Rougel and RougeLSum scores as output in Result/metric.json file.


## Recommended to run on colab

# Guide

To run on colab 

Step 1- Change python version by running the following in a cell and enter(2) to select python 3.8 (or anything less then 3.9) 

!sudo update-alternatives --config python3

Step 2 - Run this

!python --version
!sudo apt install python3-pip

Step 3 - Install Dependencies by running this

!pip install fire
! pip install transformers==4.1.0
!pip install sentencepiece 
!pip install pytorch-lightning==1.0.4
! pip install datasets
! pip install rouge_score
! pip install GitPython
! pip install rouge_score
! pip install sacrebleu
! pip install protobuf==3.20.*


Step 4 - Change the data_dir variable to give path of you files.

"data_dir": "your path to the data files folder",


Step 5 - Run the distillation.py file by running this 

! python /path/to/your/dir/distillation.py 

## Credits
We used some of the approach from the following repository:
https://github.com/huggingface/transformers/blob/e2c935f5615a3c15ee7439fa8a560edd5f13a457/examples/seq2seq

