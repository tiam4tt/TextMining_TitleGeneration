# Scientific Title Generation using transformer-based seq2seq models

## About the project
This project ultilitzes transformer-based seq2seq models from [Hugging Face](https://huggingface.co/) to generate titles for a given abstract of a scientific publication, preferably in the field of machine learning. The project was assigned as a part of the course "Text Mining" at the University of Science - VNUHCM.

## Team members

| Name | Student ID |
| --- | --- |
| Khuu Thanh Thien | 22127396 |
| Huynh Thien Thuan | 22127407 |
| Huynh Nhat Nam | 22127282 |

## Process of the project

### 1. Data collection
The data source in this project is obtained from **Springer Journal**.
All data are scrapped by team members using Python `requests` for fetching pages containing target data and `BeautifulSoup` to parse the HTML content, extracting 3 main fields:
- URL (link to the article for validity checking).
- Abstract (the summarization of that paper).
- Title (the title of that paper).

### 2. Data preprocessing
- Originally the data has over 5M data points, in which we extracted publications that are related to the field machine learning by specifying keywords and looking for the presence of these keywords in the `abstract`.
- We came down to ~91k data points after filtering. However during the process of Exploratory Data Analysis (EDA), we found out that there were few data points whose number of tokens of `abstract` exceeds 512, which is the maximum length that most of the transfomer-based models can handle. Therefore, we further filtered the data to only include data points whose tokens of `abstract` is in the range of 128 to 512 and number of tokens in `title` is between 8 and 32.
- Further preprocessing of the vocabulary (unique words and their count of occurence) reveals that using traditional scikit-learn's `train_test_split()` method does not evenly distribute the data, due to the fact that there are too many words from `test` set and `validation` set that are not present in the `train` set. This will cause the model to perform badly. Therefore, we decided to use **stratified sampling** to evenly distributing the data. 
    - The idea of **stratified sampling** is to divide data into subgroups based on their keywords (cnn, rnn, deep learning, etc.) and then sample from each subgroup to form the `train`, `validation`, and `test` set.
### 3. Model selection
For learning purposes, we tried training the data with 2 traditional seq2seq models:
- Long-Short Term Memory (LSTM) model.
- Gated Recurrent Unit (GRU) model.

Of course, none of which gave a satisfactory result. These models are just purely for studying purposes.
We then moved on to using transformer-based models from Hugging Face. Initially we experimented on various models like `facebook/bart-base`, `openai-community/gpt2`, etc. Eventually narrowed down to 2 models:
- `google/flan-t5-base`
- `google/pegasus-xsum`

Why we chose these 2 models:
- FLAN-T5 is a model that was further fine-tuned based on T5, for the same number of parameters (248M). It is trained on a large-scale mixture of tasks, including summarization, translation, and question-answering.
- Pegasus-XSum is a model that was fine-tuned on the XSum - Extreme Sumarization dataset, which is a dataset for summarization. Hence, it is optimized for the task of summarization in this project.
### 4. Model training
Training is carried out on **Kaggle** with the help of their **Tesla P100-PCIE-16GB** GPU.
(We are still in the process of training the model)
### 5. Model evaluation
### 6. Deploytment