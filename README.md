# Example of using anotherspdnet to train a SPDnet model

This repo is a small example to use [anotherspdnet](https://github.com/AmmarMian/anotherspdnet) for training.

## Installation

To help with installation, we use [just](https://github.com/casey/just) task runner, so make sur to install that first. Then, depending on your setup (cpu or gpu), two tasks are available:
* CPU: `just create_env_cpu`
* GPU: `just create_env_gpu`

This will create a conda environment called either `anotherspdnet-cpu` or `anotherspdnet-gpu`. This comes with the latest code of `anotherspdnet` grom Github. In case you need to tinker with the code, the `anotherspdnet` repo is a submodule of this one so you can download submodules:
* download code `just get_anotherspdnet`
* install locally `just install_anotherspdnet` to have the changes in the installed package


## Getting the datasets


| Dataset         | Command                      |
|-----------------|------------------------------|
| AFEW           | `just download_afewspd`      |
| SP_marti       | `just download_spmarti`      |
| HDM05          | `just download_hdm05spd`     |



## Running experiments

### AFEW dataset 

**This experiment is with the covariance data provided by the authors of the original SPDnet paper which isn't quite clear on all the steps on the pre-processing.**

You can run:
```console
just run_afew
```
which is just a fancy way to run: `python experiments/train_afew.py --storage_path results/afew/`. Lots of hyperparameters are tunable from the command-line when running the script directly:

```console
> python experiments/train_afew.py --help
usage: Training on AFEW dataset with spdnet [-h] [--hd HD] [--lr LR] [--eps EPS] [--softmax SOFTMAX] [--reeig_bias REEIG_BIAS] [--batchnorm BATCHNORM]
                                            [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--device DEVICE] [--dtype DTYPE] [--train_percentage TRAIN_PERCENTAGE]
                                            [--seed SEED] [--dataset_path DATASET_PATH] [--storage_path STORAGE_PATH]

options:
  -h, --help            show this help message and exit
  --hd HD               Hidden dimensions of spdnet
  --lr LR               Learning rate
  --eps EPS             Epsilon for SPDNet
  --softmax SOFTMAX     Use softmax activation function
  --reeig_bias REEIG_BIAS
                        Use reeig with bias term
  --batchnorm BATCHNORM
                        Use batch normalization
  --batch_size BATCH_SIZE
                        Batch size
  --epochs EPOCHS       Number of epochs
  --device DEVICE       Device (cpu or cuda)
  --dtype DTYPE         Data type (float32 or float64)
  --train_percentage TRAIN_PERCENTAGE
                        Percentage of training data
  --seed SEED           Random seed
  --dataset_path DATASET_PATH
                        Path to AFEW dataset
  --storage_path STORAGE_PATH
                        Path to save results
```


Just to verify the `anotherspdnet` against other implementations, we also provide a script:
```console
just run_afew_kobler
```
which maps to:
```console
python experiments/train_afew_kobler.py
```

and use [TSMNet](https://github.com/rkobler/TSMNet) implementation of SPDNet (as of 11 Dec 2024).


### SP_marti

An additional dataset on Stock markets obtained from [Marti](https://marti.ai/ml/2021/01/22/spdtorchnet.html) can be trained using:
```console
just run_spmarti
```

Or using the script directly, which allows to change hyperparameters:
```console
> python experiments/train_spmarti.py --help 
usage: Training on SP_marti dataset with spdnet [-h] [--hd HD] [--lr LR] [--eps EPS] [--softmax SOFTMAX]
                                                [--reeig_bias REEIG_BIAS] [--batchnorm BATCHNORM]
                                                [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--device DEVICE]
                                                [--dtype DTYPE] [--test_percentage TEST_PERCENTAGE]
                                                [--train_percentage TRAIN_PERCENTAGE] [--seed SEED] [--shuffle_loader]
                                                [--dataset_path DATASET_PATH] [--storage_path STORAGE_PATH]

options:
  -h, --help            show this help message and exit
  --hd HD               Hidden dimensions of spdnet
  --lr LR               Learning rate
  --eps EPS             Epsilon for SPDNet
  --softmax SOFTMAX     Use softmax activation function
  --reeig_bias REEIG_BIAS
                        Use reeig with bias term
  --batchnorm BATCHNORM
                        Use batch normalization
  --batch_size BATCH_SIZE
                        Batch size
  --epochs EPOCHS       Number of epochs
  --device DEVICE       Device (cpu or cuda)
  --dtype DTYPE         Data type (float32 or float64)
  --test_percentage TEST_PERCENTAGE
                        Percentage of final test data.
  --train_percentage TRAIN_PERCENTAGE
                        Percentage of training data
  --seed SEED           Random seed
  --shuffle_loader      Whether to shuffle data loaders at each epoch.
  --dataset_path DATASET_PATH
                        Path to AFEW dataset
  --storage_path STORAGE_PATH
                        Path to save results
```

### HDM05 dataset

**This experiment is with the covariance data provided by the authors of the original SPDnet paper which isn't quite clear on all the steps on the pre-processing.**

You can run:
```console
just run_hdm05
```
which is just a fancy way to run: `python experiments/train_hdm05.py --storage_path results/hdm05/`. Lots of hyperparameters are tunable from the command-line when running the script directly:
```console
> python experiments/train_hdm05.py --help
usage: Training on SP_marti dataset with spdnet [-h] [--hd HD] [--lr LR] [--eps EPS] [--softmax SOFTMAX]
                                                [--reeig_bias REEIG_BIAS] [--batchnorm BATCHNORM]
                                                [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--device DEVICE]
                                                [--dtype DTYPE] [--test_percentage TEST_PERCENTAGE]
                                                [--train_percentage TRAIN_PERCENTAGE] [--seed SEED] [--shuffle_loader]
                                                [--dataset_path DATASET_PATH] [--storage_path STORAGE_PATH]

options:
  -h, --help            show this help message and exit
  --hd HD               Hidden dimensions of spdnet
  --lr LR               Learning rate
  --eps EPS             Epsilon for SPDNet
  --softmax SOFTMAX     Use softmax activation function
  --reeig_bias REEIG_BIAS
                        Use reeig with bias term
  --batchnorm BATCHNORM
                        Use batch normalization
  --batch_size BATCH_SIZE
                        Batch size
  --epochs EPOCHS       Number of epochs
  --device DEVICE       Device (cpu or cuda)
  --dtype DTYPE         Data type (float32 or float64)
  --test_percentage TEST_PERCENTAGE
                        Percentage of final test data.
  --train_percentage TRAIN_PERCENTAGE
                        Percentage of training data
  --seed SEED           Random seed
  --shuffle_loader      Whether to shuffle data loaders at each epoch.
  --dataset_path DATASET_PATH
                        Path to HDM05 dataset
  --storage_path STORAGE_PATH
                        Path to save results
```

## Author

Ammar Mian, [ammar.mian@univ-smb.fr](mailto:ammar.mian@univ-smb.fr).
