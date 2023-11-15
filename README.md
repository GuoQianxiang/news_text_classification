# News_Text_Classification
- Hello！Welcome all friends who enjoy machine learning, deep learning, and natural language processing!!!

- Taking the data in the data directory as an example, we conducted natural language modeling, trained and compared different models.

- The processing structure is as follows：
## Preprocess
### 1、clean data
- drop blank rows
- stop words
- split useless letters
### 2、Vectorize sentences
#### 2.1 vectorize word：
- Word2vec
- Doc2vec
- TF-IDF
- Glove
#### 2.2 calculate sentence：
- sum
- mean
- weighted mean
## Model
You can use package in sklearn or pytorch\tensorflow.
### 1、CNN
### 2、KNN
### 3、MLP
### 4、NBS

## Evaluation
- Use F1-score\accuracy\precision\revision to evaluate the model.
- You should preload the data by dataloader in /utils/dataloader.
## Utils
### dataloader
- One-hot and word2vec are different after being converted to dataframe type.
- You should preprocess if you want to use converted_data.
- You can continue to model regardless of dataloader if you training after preprocessing.

### utils
- shuffle data and split data by 80%/20%.
- To be updating...