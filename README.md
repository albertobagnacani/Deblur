# Scale-recurrent Network for Deep Image Deblurring
For the project details, see `Relation_Bagnacani.pdf`

[comment]: <> (
# Download the project
```
git clone https://github.com/albertobagnacani/Deblur.git
```
)

# Deployment
Run
```
conda env create -f environment.yml
```
to install the needed dependencies.

Activate the environment:
```
conda activate deblur
```

# Dataset preparation
## CIFAR-10
Create the directory that will contain the dataset:
```
mkdir -p res/datasets/cifar-10/
```

Download the dataset and unzip it: 
```
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xf cifar-10-python.tar.gz
```

Move the downloaded files inside the dataset folder:
```
mv cifar-10-batches-py/* res/datasets/cifar-10/
```

## REDS
Create the directory that will contain the dataset:
```
mkdir -p res/datasets/reds/
```

Download the following files from https://seungjunnah.github.io/Datasets/reds.html and then unzip them:
- train_sharp
- val_sharp
- train_blur
- val_blur

Move the folders inside the dataset folder:
```
mv train_sharp res/datasets/reds/train
mv train_blur res/datasets/reds/train
mv val_sharp res/datasets/reds/val
mv val_blur res/datasets/reds/val
```

# Download weights
Create the directories that will contain the weights:
```
mkdir -p res/models/reds/
mkdir -p res/models/cifar/
```

Download the weights from here (~ 237 MB each):
- REDS: https://drive.google.com/file/d/1nlkpi-Q3U7RGKczhXdrwtbt-HbkNHItU/view?usp=sharing
- CIFAR-10: https://drive.google.com/file/d/1hx1-D4ogGWjbXjdBDZvo7RsANlFbCnl_/view?usp=sharing

Move the downloaded files in the following paths (be aware to respect this naming): 
- REDS: `res/models/reds/model-reds-100.h5`
- CIFAR-10: `res/models/cifar/model-cifar-100.h5`

# Run
```
python3 src/main.py
```
A subset of the (hyper)parameters are defined in the `src/params.json` file. 
Note that this file is actually used by the `src/main.py` script.

Note that the first run may need some time to prepare the datasets in the structure needed by the project.

## params.json file
The `src/params.json` contains a list of (hyper)parameters to run the script:
- "task": "cifar" or "reds". Dataset on which perform the training/prediction/evaluation
- "epochs": int. Number of epochs to train
- "batch_size": int. Batch size (on NVIDIA GEFORCE RTX 2060, batch size = 8 for the REDS task and batch size = 32 
for the CIFAR task (up to 512 in this case))
- "initial_lr": float. Initial Learning Rate of the Neural Network
- "load_epoch": int. Epoch to load to resume training or making prediction/evaluation. If 0, it doesn't load any 
model/weights (new training)
- "action": int. Can be 0, 1 or 2 for training, predicting and evaluating the Neural Network, respectively
- "subset": boolean. true if the training/validation set must be a subset of the original one (for fast testing. Note
that you have to create a subset manually. This option is usually left to false); 
false otherwise
- "seed": int. Seed
- "mc_period": int. ModelCheckpoint saving period (frequency in epochs of the the model/weights saving). If set to 1, 
the model/weights are saved at each epoch

### Examples for params.json file
For training a new model, set:
- "load_epoch": 0 (no checkpoints to be resumed, new training)
- "action": 0

For predicting, set:
- "load_epoch": 100 (the model file must be named `model-reds-100.h5` (or `model-cifar-100.h5` for CIFAR-10 task) 
if you want to load the model trained after e.g. 100 epochs)
- "action": 1

For evaluating, set:
- "load_epoch": 100
- "action": 2