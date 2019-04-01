EDIT-TRANSFORMER
==================

This project introduce the "edit-transformer". A generative model based on the Google Transformer 
seq-to-seq from "Attention is all you need" that enables data augmentation. The data augmentation is performed by 
editing existing prototype sentences in a similar way as in "Generating sentences by editing prototypes". Training on 
pair of similar sentences allows the model to learn how to edit sentences and at inference performs generation.
 
In this github repository, a few scripts are made available to easely train and generate new data using the 
"edit-transformer" model.

### Pre-requisites

This code requires python version 3.7.2. All other dependencies and requirements as well as the
environements used during this work can be reproduced using the provided Dockerfile. In order to do so, Docker must be
installed and if using GPU (highly recommended for training) nvidia-docker is also required. The training dataset is 
also coming from a separate source and must be downloaded. An example dataset of yelp review is provided and can be
found [here](https://worksheets.codalab.org/bundles/0x99d0557925b34dae851372841f206b8a/).

### Building the environment

In order to reproduce the environment used in this project first download the dataset specified in pre-requisites and unzip it in a 
```data``` folder. Then clone this repository in another folder using: 

```bash
git clone --recursive https://github.com/grll/edit-transformer.git
```

> Recursive is mandatory to include the submodules (torchtext in this case).

The folder structure aimed after these steps should look like the following:

    .
    |-- edit-transformer
        |-- ...
    |-- data
        |-- datasets
            |-- yelp
                |-- train.tsv
                |-- test.tsv
                |-- valid.tsv
                |-- free.txt
 
 Once this structured is obtained, the docker image corresponding to this project environment can be built using the 
 following commands:
 
```bash
cd edit-transformer
docker build --build-arg UNAME=$(whoami) --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t edit-transformer:0.0.2 .
```

> The build-arg allows to use the current user inside the docker automatically avoiding root created content from inside
 the docker image.

Once the image is built, it is possible to access the environement in interactive mode using the following command:

```bash
docker run -it --rm -v $(pwd):/code -v $(pwd)/../data:/data -e INTERACTIVE_ENVIRONMENT=True edit-transformer:0.0.2 /bin/bash
```

> It will run a temporary bash terminal inside the container and mount the current folder in the `/code` folder of 
the container and the data folder in the `/data` folder of the container.

### Preprocessing

Before training the model the data downloaded must be briefly preprocessed using the `edit_transformer/preprocess.py` 
script. This script format the data and filter insertions and deletions between the pair of sentences using the 
`free.txt` file if specified so in its config. The corresponding config file can be found in :
`configs/edit_transformer/preprocess.txt`.

This preprocessing step can be directly ran using the following command:

```bash
docker run -d --rm -v $(pwd):/code -v $(pwd)/../data:/data -e CUDA_VISIBLE_DEVICES=0 edit-transformer:0.0.2 python3.7 -u edit_transformer/preprocess.py
```

> In case of error make sure everything is correct in the corresponding config file 
(`configs/edit_transformer/preprocess.txt`).
>
> This will run the script in detached mode (in the background). Output will be found in the newly created `logs` folder
inside the `data` folder.

### Training

Once the preprocessing step is finished, the model can be trained using the 3 generated files. The training script 
available in `edit_transformer/training.py` allows to train the "edit-transformer" using the config specified in 
`configs/edit_transformer/edit_transformer.txt`. Once again make sure everything is correct (espacially the paths) in 
this file before running the following command:

```bash
docker run -d --rm -v $(pwd):/code -v $(pwd)/../data:/data -e CUDA_VISIBLE_DEVICES=0 edit-transformer:0.0.2 python3.7 -u edit_transformer/training.py
```

> This script generates a new folder for each run with a number in `/data/training_runs`. In this specific folder you 
will find the logs of the training run, its config, some output examples from the training and the tensorboard log 
events.

### Generation

In order to use the model in generation mode, the configuration of a finished training run must be appropriately 
provided through the config file in `configs/edit_transformer/generation.txt`. In particular, all the `model` config
key must be set appropriately in order for the generation script to find the output of your training run.

Once this is done an example generation script allows you to input a sentence and returns you 5 generated samples from
the model. In order to run this script you can run the following command:

```bash
docker run -it --rm -v $(pwd):/code -v $(pwd)/../data:/data -e INTERACTIVE_ENVIRONMENT=True edit-transformer:0.0.2 python3.7 edit_transformer/generation.py
```

> Note that you can tune the time taken by the script to perform the beam search by reducing either q_limit the maximum
node of beam search per sample or attempts the number of beam search to run with different edit vector sampled in the 
script config previously mentioned.

A generated output will look like the following:

![Generation Output](/images/output.png)

where each generated sample correspond to one row and the first numbered column corresponds to the probability output of
 the model for the generated sequence and the second numbered column is the corresponding normalized score used in the 
beam search.
