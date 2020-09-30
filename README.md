# Scale-recurrent Network for Deep Image Deblurring
For the project details, see `Relation_Bagnacani.pdf`

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
A subset of the (hyper)parameters are defined in the `params.json` file. 
Note that this file is actually used by the `main.py` script.

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