# Encoder-Decoder GRU framework

This project presents a training and evaluation strategy that outperforms the original publication [DOI: 10.1016/j.est.2022.106461](https://doi.org/10.1016/j.est.2022.106461). The evaluation can be replicated using the provided notebook in `Evaluation/calculate_metrics.ipynb`.

## Installation

To install the python version 3.8.18 and setup the virtual environment using [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation):
```sh
pyenv install 3.8.18
pyenv virtualenv 3.8.18 rnn_gru
pyenv activate rnn_gru
```
To install the dependencies, use the following command from the main directory:

```sh
pip install -r requirements.txt
``` 

## Setup Weights and Biases
All the hyperparameters and metrics are logged to Weights and Biases by default.  
To get started with Weights and Biases, create an account on [Weights and Biases](https://wandb.ai/).
For authorization during first training run, an API key is required which can be acquired from https://wandb.ai/authorize  

Documentation for Weights and Biases : https://docs.wandb.ai/

## Training the Model

To train the model, use the following command from the main directory:
```sh
python scripts/train.py epochs=10 device=gpu
```
Arguments for the model and data can be passed directly via the command line. For example:
```sh
python scripts/train.py epochs=5 method.stride=16 method.decoder_input_length=1980 lr=0.0001
``` 
Additional arguments can be found in `config/method/rnn.yaml` and `config/train.yaml`.

### Note:

* If wandb is not already logged in, the script will ask to sign-in/authorize using the API key on https://wandb.ai/authorize before starting the training process.  
* In addition to the automatically generated plots on wandb, custom line plots can be added. For ex: to visualize profile 1's accuracy over epoch, create a new panel and select a [line plot](https://docs.wandb.ai/guides/app/features/panels/line-plot). Then set X-axis as `epoch` and Y-axis as `profile_1_mae`.

## Running on CPU

The scripts are designed to run via the command line from the main directory and on GPU. To run the model on the CPU, comment out the callback function `on_validation_epoch_end()` in `src/RNN/models.py` and pass the argument `device=cpu`

## Run on a HPC System
* First, setup a virtual environment. The documentation to set up a custom virtual environment on TUD HPC cluster can be found here : https://compendium.hpc.tu-dresden.de/software/python_virtual_environments/?h=envir#python-virtual-environments
* To run the script as a SLURM job, use the bash script `alpha-train.sh`. Update the path to the virtual environment before running the script.
* After updating the path to the environment, to submit a job:
```sh
 sbatch alpha-train.sh
```
## Training Logs

Since [hydra](https://hydra.cc/docs/intro/) was used in the project, training checkpoints along with the corresponding hyperparameters and configurations are saved locally in the main directory in the path `outputs/$day$/$time$`  
By default, model checkpoints are saved every epoch since `save_top_k=-1`. To save for ex: 5 checkpoints with best val_loss, use argument `save_top_k=5`. 

## Outlook
* Although the model performs well on the dataset, it needs to be compared to a public benchmark dataset to compare different implementations in the literature.
* Performance comparison between GRU and LSTM implementations.
* There is potential for improvements in the data preprocessing steps.
