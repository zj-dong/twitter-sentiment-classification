# Twitter Sentiment Analysis - SentimentalSmarts

This repository contains code used for the course project of the Computational Intelligence Lab at ETH (FS 2019).

## Reproducing results

We only describe in detail how to reproduce our final submission on Kaggle using the code in this repository. 
The other methods described in the submitted project report can be reproduced in similar ways using the parameters
given in the report.

**NOTE:** Unless stated otherwise the commands are assumed to be executed from the `src` directory and data is assumed
to be stored in the `data` directory (needs to be created).

### Packages

We use Python 3.6.4 for our implementation. The required packages can be installed using the provided `requirements.txt` 
file. Note that this will install the GPU version of TensorFlow.

### Data preprocessing

#### Data formatting

For the biGRU-CNN model and the Transformer models, we first organise the data in TSV files. With `DATA_DIR` being the
directory where the downloaded data from Kaggle (separated in positive and negative sentiment) is stored, the data will be 
concatenated and properly formatted using the following command:
```bash
python --output_path ../data/ data/process_data.py DATA_DIR
```
Several flags can be set to perform preprocessing on the data, but this is not used for the described models.

#### Separation into training and validation sets

While it is possible to load the complete data and split it into training and validation sets using our implementation,
we used well-defined training and validation sets for the sake of reproducibility. To do this a directory to save the
split data in should be created (`mkdir ../data/split_data/`)
```bash
python models/validation.py --data_path ../data/train_data_full.tsv --down_sample_val 0.1 --down_sample_train 0.1 --save_path ../data/split_data/ 
```
Note that the `--save_indices` flag can be used to save the indices of the data as well as the data itself. Thus it is
possible to do other preprocessing on the data (e.g. for the LSTM and biLSTM models) without shuffling it and still use
the same training and validation sets.

### Training

Due to time limitations the final biGRU-CNN model was trained for 7 epochs on our full training set (2 million tweets)
and for an 8th epoch on only 75% of that data (1.5 million tweets). For the embedding any pre-trained vectors which are
stored in plain text using the format `token value1 value2 value3 ...` can be used. We use 200-dimensional GloVe 
embeddings pre-trained on a large Twitter corpus, which can be downloaded from [here](http://nlp.stanford.edu/data/glove.twitter.27B.zip).
With `EMBEDDING_FILE` being the file path to the correct embedding file and the data being stored in `data`, the first 
7 epochs of training are performed as follows:
```bash
python models/cnn_rnn.py --embedding_file EMBEDDING_FILE --vocabulary_size 10000 --vocabulary_save_file ../data/bi_gru_cnn_vocab_10000.pkl --model_name bi_gru_cnn --batch_size 16 --epochs 7 --learning_rate 0.0001 --filters 512 --kernel_size 5 --pool_size 2 --cell_type GRU --cell_size 200 --cell_stack_size 2 --rnn_first --train_val_split 1.0 --save_every 1 --checkpoint_dir ../checkpoints/ --log_dir ../logs/ --validation_data ../data/split_data/validation_data_labeled_ds01.tsv ../data/spit_data/training_data_full.tsv
```

To continue training using only 75% of the training data:
```bash
python models/cnn_rnn.py --embedding_file EMBEDDING_FILE --vocabulary_size 10000 --vocabulary_source_file ../data/bi_gru_cnn_vocab_10000.pkl --max_length 130 --model_name bi_gru_cnn_continuation --model_load_file ../checkpoints/bi_gru_cnn/model-007.hdf5 --batch_size 16 --epochs 1 --learning_rate 0.0001 --train_val_split 1.0 --train_on_subset 0.75 --save_every 1 --checkpoint_dir ../checkpoints/ --log_dir ../logs/ --validation_data ../data/split_data/validation_data_labeled_ds01.tsv ../data/spit_data/training_data_full.tsv
```

### Prediction

The final predictions were done by averaging the probabilities of the positive class from the last 3 saved models.
A directory `data/voting/` should be created (`mkdir ../data/voting` from the `src` directory) and then the following
commands should be run to make all 3 predictions.

Predictions for the model after 6 epochs:
```bash
python models/cnn_rnn.py --embedding_file EMBEDDING_FILE --vocabulary_size 10000 --vocabulary_source_file ../data/bi_gru_cnn_vocab_10000.pkl --max_length 130 --model_load_file ../checkpoints/bi_gru_cnn/model-006.hdf5 --predict --prediction_save_file ../data/voting/predictions-6-ep.csv --predict_probabilities --batch_size 16 --log_dir ../logs/ ../data/test_data.tsv
```

Predictions for the model after 7 epochs:
```bash
python models/cnn_rnn.py --embedding_file EMBEDDING_FILE --vocabulary_size 10000 --vocabulary_source_file ../data/bi_gru_cnn_vocab_10000.pkl --max_length 130 --model_load_file ../checkpoints/bi_gru_cnn/model-007.hdf5 --predict --prediction_save_file ../data/voting/predictions-7-ep.csv --predict_probabilities --batch_size 16 --log_dir ../logs/ ../data/test_data.tsv
```

Predictions for the final trained model after "7.75" epochs:
```bash
python models/cnn_rnn.py --embedding_file EMBEDDING_FILE --vocabulary_size 10000 --vocabulary_source_file ../data/bi_gru_cnn_vocab_10000.pkl --max_length 130 --model_load_file ../checkpoints/bi_gru_cnn_continuation/model-001.hdf5 --predict --prediction_save_file ../data/voting/predictions-8-ep.csv --predict_probabilities --batch_size 16 --log_dir ../logs/ ../data/test_data.tsv
```

Since the `voting.py` script averages the probabilities from all files in the directory the class predictions (using
labels -1 and 1) should be moved out of the directory:
```bash
mv ../data/voting/predictions-*-ep.csv ../data/
```

Now the soft voting can be done:
```bash
python models/voting.py --soft_voting ../data/voting/
```

The resulting predictions are stored in `data/voting/soft_voting_predictions_3_clf.csv`.

### Evaluation on validation data

To evaluate the averaged predictions on the validation data, the steps in the previous section should be followed, 
replacing `../data/test_data.tsv` with `../data/split_data/validation_data_unlabeled_ds01.tsv` and changing the output 
directory from `../data/voting/` to `../data/voting_validation/`. To then obtain the accuracy of the predictions on the 
validation set, the following should be run:
```bash
python models/validation.py --data_path ../data/split_data/validation_data_labeled_ds01.tsv --evaluate --prediction_path ../data/voting_validation/soft_voting_predictions_3_clf.csv
```

## Other models

### Logistic Regression

We followed the steps described [here](https://github.com/dalab/lecture_cil_public/blob/master/exercises/ex6) to 
obtain `data/vocab.txt`, `data/vocab.pkl`, which record the token's frequency counts and its corresponding index in 
the defined vocabulary respectively. Word embeddings should be specified in `EMBEDDING_FILE`. If pre-trained GloVe 
embeddings are used, the `--pre_trained` flag should be set as well.

Then, apply grid search with cross validation to best parameter on 10% of the training data using word embeddings:
```bash
 python baselines/classification.py --verbosity 100 --classifier lr --params C:0.1,1,10 --grid_search --n_jobs 16 --save_file ../checkpoints/glove_lr ../data/split_data/training_data_small_ds01.tsv EMBEDDING_FILE ../data/vocab.txt ../data/vocab.pkl
```

Now use the best penalty parameter `C = 1` on the full training data:
```bash
python baselines/classification.py ---verbosity 100 --classifier lr --params C:1 --n_jobs 16 --save_file ../checkpoints/lr_C_1 ../data/split_data/training_data_full.tsv EMBEDDING_FILE ../data/vocab.txt ../data/vocab.pkl --pre_trained
```

Finally, restore the trained model and test on the unlabeled validation data. The prediction is stored under `lr_validation_res.csv`.
```bash
 python baselines/classification.py --load_file ../checkpoints/lr_C_1.pkl --predict --prediction_save_file lr_validation_res ../data/split_data/validation_data_unlabeled_ds01.tsv EMBEDDING_FILE ../data/vocab.txt ../data/vocab.pkl
```

### Transformer networks

Training and prediction for the Transformer networks are essentially done in the same way as for the RNN-CNN model.
Important to note is that we still implemented models which can be pre-trained on other data sets. Thus the flag
`--fine_tune_model` should be set to match the results presented in the project report. There is information about all
other parameters available from the scripts' help page.

**NOTE:** When doing predictions using the Transformer models a lot of memory seems to be used. This may be due to the
implementation of some custom metrics used for the training of the models. Thus prediction either has to be done with a
very large amount of RAM or the flag `--predict_in_splits` should be set, which will split the data into chunks and
combine the predictions for each chunk.

### LSTM and biLSTM

To prepare the training data for the LSTM and biLSTM models, the following should be run:
```bash
awk '{printf("1,  %s\n", $0)}' train_pos_full.txt > sentiment_train_pos_full.txt
awk '{printf("0,  %s\n", $0)}' train_neg_full.txt > sentiment_train_neg_full.txt
cat sentiment_train_pos_full.txt sentiment_train_neg_full.txt > sentiment_train.txt
awk '{printf("%06d, %s\n", NR, $0)}' sentiment_train.txt > full_train.txt
```

The LSTM model used for the baseline can be found [here](https://github.com/abdulfatir/twitter-sentiment-analysis/blob/master/code/lstm.py).
For both the LSTM and biLSTM models, the data should be preprocessed using [this script](https://github.com/abdulfatir/twitter-sentiment-analysis/blob/master/code/preprocess.py), 
and the frequency distributions of tokens should be computed using [this script](https://github.com/abdulfatir/twitter-sentiment-analysis/blob/master/code/stats.py).

Once the processed data and the token frequencies have been saved, the models can be trained and the `predict.py` script
can be used to make predictions.

## Code from other sources

For training GloVe embeddings ourselves on the provided data we adapted the solution to one of the course exercises that
can be found [here](https://github.com/dalab/lecture_cil_public/blob/master/exercises/ex6/glove_solution.py).

For the LSTM and biLSTM models we used code from [this repository](https://github.com/abdulfatir/twitter-sentiment-analysis) 
on the topic of Twitter sentiment analysis with some minor changes.
