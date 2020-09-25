# Scale-recurrent Network for Deep Image Deblurring
For the project details, see `Relation_Bagnacani.pdf`

# Deployment
Run
```
conda env create -f environment.yml
```
to install the needed dependencies.

# Run
```
python3 src/main.py
```
The (hyper)parameters are defined in the `params.json` file or in the beginning of the `main.py` script 
(more complete list).

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
- "subset": boolean. true if the training/validation set must be a subset of the original one (for fast testing); 
false otherwise
- "seed": int. Seed
- "mc_period": int. ModelCheckpoint saving period (frequency in epochs of the the model/weights saving). If set to 1, 
the model/weights are saved at each epoch