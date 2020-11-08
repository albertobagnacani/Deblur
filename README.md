# Deep Image Deblurring
For the project details, see:
- Relation: https://docs.google.com/document/d/1LiOAaUSIEWBpxRx6_HTnsfB5-Y9ZCqZPKQxvQJNx9RQ/edit?usp=sharing
- Presentation: https://docs.google.com/presentation/d/1uKaZ98-PazS-LcbIWVa0RbZZUyG4oiORhSBVfOCI5so/edit?usp=sharing

# Download the project
```
git clone https://github.com/albertobagnacani/Deblur.git
```
and cd it:
```
cd Deblur/
```

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
Create the directories that will contain the dataset:
```
mkdir -p res/datasets/REDS/train/train_sharp
mkdir -p res/datasets/REDS/train/train_blur
mkdir -p res/datasets/REDS/val/val_sharp
mkdir -p res/datasets/REDS/val/val_blur
```

Download the following files from https://seungjunnah.github.io/Datasets/reds.html and then unzip them:
- train_sharp
- val_sharp
- train_blur
- val_blur

Move the folders inside the dataset folder:
```
mv train_sharp/* res/datasets/REDS/train/train_sharp
mv train_blur/* res/datasets/REDS/train/train_blur
mv val_sharp/* res/datasets/REDS/val/val_sharp
mv val_blur/* res/datasets/REDS/val/val_blur
```

The structure should be:
```
res/datasets/reds/train/train_sharp/000
res/datasets/reds/train/train_sharp/001
...
```

# Download models
Create the directories that will contain the models:
```
mkdir -p res/models/reds/checkpoints/
mkdir -p res/models/cifar/checkpoints/
```

Download the models from here:
- REDS-srn: https://drive.google.com/file/d/1nlkpi-Q3U7RGKczhXdrwtbt-HbkNHItU/view?usp=sharing
- CIFAR-10-srn: https://drive.google.com/file/d/1hx1-D4ogGWjbXjdBDZvo7RsANlFbCnl_/view?usp=sharing
- CIFAR-10-fcn: https://drive.google.com/file/d/1JNxLsSgLPNn_diqfMNTJHNBbNY-o-Qu6/view?usp=sharing
- CIFAR-10-unet: https://drive.google.com/file/d/1EbckBTXt0N3KS5yFVR4eT9GXzzM5lyI7/view?usp=sharing
- CIFAR-10-rednet: https://drive.google.com/file/d/1thLvmVehKwrccgdJlBYUx10wmn3tyrjO/view?usp=sharing

Move the downloaded files in the following paths (parent of the created `checkpoints` folders; be aware to respect this 
naming): 
- REDS-srn: `res/models/reds/`
- CIFAR-10-(srn/fcn/unet/rednet): `res/models/cifar/`

The structure should be:
```
res/models/reds/model-reds-srn-100.h5
res/models/cifar/model-cifar-srn-100.h5
res/models/cifar/model-cifar-fcn-100.h5
res/models/cifar/model-cifar-unet-100.h5
res/models/cifar/model-cifar-rednet-100.h5
```

# Run
Enter the `src` folder:
```
cd src
```
and run the main:
```
python3 main.py
```

A subset of the (hyper)parameters are defined in the `src/params.json` file. 
Note that this file is actually used by the `src/main.py` script.

Note that the first run may need some time to prepare the datasets in the structure needed by the project.

## params.json file
The `src/params.json` contains a list of (hyper)parameters to run the script:
- "task": "cifar" or "reds". Dataset on which perform the training/prediction/evaluation
- "model": "srn" or "fcn" or "unet" or "rednet". The architecture to load
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

The REDS test set (the downloaded validation set is actually used as a test set) predictions can be found in the 
`res/datasets/REDS/out/val/folder/` folder.

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
