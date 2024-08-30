# RadCloud

Official Git Repository for the Duke Cyber Physical Systems Laboratory (CPSL) RadCloud project. For more details see our website at [https://sites.google.com/view/radcloudduke/home](https://sites.google.com/view/radcloudduke/home)

## Installation
In order for the code to work properly, the following steps are required
1. Install correct version of python
2. Install RadCloud using Poetry

### 1. Setup Python environment

You can install the correct version of python using the deadsnakes PPA or via Anaconda

#### Deadsnakes PPA (requires sudo access)
1. On ubuntu systems, start by adding the deadsnakes PPA to add the required version of python.
```
sudo add-apt-repository ppa:deadsnakes/ppa
```

2. Update the package list
```
sudo apt update
```

3. Install python 3.8 along with the required development dependencies
```
sudo apt install python3.8 python3.8-dev
```

The following resources may be helpful [Deadsnakes PPA description](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa), [Tutorial on Deadsnakes on Ubuntu](https://preocts.github.io/python/20221230-deadsnakes/)

#### Conda (Backup)
1. If conda isn't already installed, follow the [Conda Install Instructions](https://conda.io/projects/conda/en/stable/user-guide/install/index.html) to install conda
2. Use the following command to download the conda installation (for linux)
```
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
```
3. Run the conda installation script (-b for auto accepting the license)
```
bash Anaconda3-2023.09-0-Linux-x86_64.sh -b
```
3. Once conda is installed, create a new conda environment with the correct version of python
```
conda create -n RadCloud python=3.8
```

### 2. Clone the RadCloud repository
```
git clone https://github.com/davidmhunt/radcloud.git
```

### 3. Install RadCloud using Poetry

#### Installing Poetry:
 
1. Check to see if Python Poetry is installed. If the below command is successful, poetry is installed move on to setting up the conda environment

```
    poetry --version
```
2. If Python Poetry is not installed, follow the [Poetry Install Instructions](https://python-poetry.org/docs/#installing-with-the-official-installer). On linux, Poetry can be installed using the following command:
```
curl -sSL https://install.python-poetry.org | python3 -
```

3. After installing poetry, add the poetry path to your shell configuration. On linux systems, open your .bashrc file (located in your home directory typically), add a new line, and paste the following command (NOTE: Replace USERNAME with your username):
```
export PATH="/home/USERNAME/.local/bin:$PATH"
```

4. Finally, if you are connecting to a server remotely over SSH, run the following command to ensure a smooth installation process with Poetry

```
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

#### Installing RadCloud
1. Navigate to the Odometry foler (this folder) and execute the following command.

```
cd radcloud
conda activate RadCloud #(Only if you installed python using Anaconda)
poetry install
```

2. If you get an an error saying: "Failed to unlock the collection!" or experience other issues, try executing the following command in the terminal:
```
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

#### Installing Pytorch
Since many people have different systems with varying support for GPUs, we don't install a default version of pytorch. As such, following the following instructions to install the correct version of pytorch for your system.

1. Navigate to the [pytorch installation page](https://pytorch.org/get-started/locally/). Select the requirements for your system. However, under the "package" select the "Pip" option. Once you have specified the options for your system, you'll get a command similar to this
```
pip3 install torch torchvision torchaudio
```
2. Navigate to the RadCloud folder
```
cd radcloud
```
3. Start a poetry shell
```
poetry shell
```
4. run the command given by the pytorch website
```
pip3 install torch torchvision torchaudio
```
5. If this runs normally, you should now be good to exit the poetry shell
```
exit
```
## Accessing the dataset

We are working to host the dataset on another service, but for now, please email [david.hunt@duke.edu](mailto:david.hunt@duke.edu) for access to the radcloud dataset. The dataset contains 3 files: "ugv_rapid_movement_dataset.zip", "ugv_seen_dataset.zip", and "ugv_unseen_dataset.zip". Once downloaded, the dataset can be prepared using the following commands:

1. Move the folders into the [data](./data/) folder.

2. Next, either manually unzip the files, or you can use the helpful unzip_datasets.sh script as follows. NOTE: the dataset is quite large. If you instead wish to only use the test datasets, you can also choose to only unzip certain files if you prefer.
```
./unzip_datasets
```

### Using the datasets for model training and testing

To generate a dataset for training or evaluation, there are two options:

1. **Jupyter Notebooks** The dataset contains the raw radar data from the IWR1443 recorded using the DCA1000. To generate a test or training dataset from the raw data, please use the generate_dataset.ipynb jupyter notebook found in the [Notebooks](./Notebooks/) folder. Within this notebook, you can toggle between three of our primary testing and training datasets used in the RadCloud paper. Note that there is a notebook for generating datasets with both radar and lidar data or just radar data (if you have recorded your own dataset).

2. **Scripts** If you instead want to use a python script  please use the generate_dataset.py script found in the [scripts](./scripts/) folder. Within this script, you can toggle between three of our primary testing and training datasets used in the RadCloud paper. To call the script using the installed poetry environment, you can use the following terminal command
```
cd radcloud/scripts
poetry run python generate_dataset.py
```
Note that there are scripts for generating datasets with both radar and lidar data or just radar data (if you have recorded your own dataset).

### Recording your own test dataset

To record your own dataset, reach out to [david.hunt@duke.edu](mailto:david.hunt@duke.edu) for instructions on the specific radar configuration used as well as the format that the raw radar data should be stored in. The Duke CPSL maintains a ROS-compatible library called [CPSL_TI_RADAR](https://github.com/davidmhunt/CPSL_TI_Radar) which can be used to collect datasets. For instructions on how to use this tool, reach out to [david.hunt@duke.edu](mailto:david.hunt@duke.edu).

## Viewing model results

We have included a pre-trained RadCloud model in the [working_dir](./working_dir/) folder so that you don't have to train the model to get started looking at results. To view results from a trained model, please use the view_results.ipynb jupyter notebook from the [Notebooks](./Notebooks/) folder. Note, a dataset must be generated before viewing model results, but the view_results.ipynb file will automatically generate a dataset for viewing results by default. 

## Generating a movie

We have included a pre-trained RadCloud model in the [working_dir](./working_dir/) folder so that you don't have to train the model to get started looking at results. While the view_results scripts and notebooks can view a single frame, we also include two scripts that can be used to generate a movie for a given dataset. To utilize these scripts, please see the generate_movie.py file found in the [scripts](./scripts/) folder. The script will have to be modified depending on the dataset you want to generate a movie for. Once you've specified the dataset you want to use, run the following command to generate a movie:

```
cd radcloud/scripts
poetry run python generate_movie.py
```
Note that there are scripts for generating datasets with both radar and lidar data or just radar data (if you have recorded your own dataset).


## Training a model
While we provide the originally trained RadCloud model, we also include scripts that can be used to retrain or train a modified version of the RadCloud model. This can be done using one of two methods:

1. **Jupyter Notebooks** To train a new model, please use the train_model.ipynb jupyter notebook found in the [Notebooks](./Notebooks/) folder. 

2. **Scripts** If you instead want to use a python script  please use the train_model.py script found in the [scripts](./scripts/) folder. To call the script using the installed poetry environment, you can use the following terminal command
```
cd radcloud/scripts
poetry run python train_model.py
```
Note that there are scripts for generating datasets with both radar and lidar data or just radar data (if you have recorded your own dataset).
