# Encoder-Decoder GRU framework

This project presents a training and evaluation strategy that outperforms the original publication [DOI: 10.1016/j.est.2022.106461](https://doi.org/10.1016/j.est.2022.106461). The evaluation can be replicated using the provided notebook in `Evaluation/calculate_metrics.ipynb`.

## Installation

The python version used is 3.8.18. 
To install the dependencies, use the following command from the main directory:

```sh
pip install -r requirements.txt
``` 

## Training the Model

To train the model, use the following command from the main directory:
```sh
python scripts/train.py epochs=10
``` 
Arguments for the model and data can be passed directly via the command line. For example:
```sh
python scripts/train.py epochs=5 method.stride=16
``` 
Additional arguments can be found in `config/method/rnn.yaml` and `config/train.yaml`.

## Running on CPU

The scripts are designed to run via the command line from the main directory and on GPU. To run the model on the CPU, comment out the callback function `on_validation_epoch_end()` in `src/RNN/models.py`.

## Outlook
* Although the model performs well on the dataset, it needs to be compared to a public benchmark dataset to compare different implementations in the literature.
* Performance comparison between GRU and LSTM implementations.
* There is potential for improvements in the data preprocessing steps.
