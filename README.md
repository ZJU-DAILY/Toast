# Toast
Python implementation of Toast: Task-oriented Augmentation for Spatio-Temporal Data.

## Dependencies
``` shell
pip install -r requirements.txt
```

Another dependency `traj-dist` is required, and it can be install from [<u> this link </u>](https://github.com/bguillouet/traj-dist).

## Datasets
The dataset `T-drive` can be downloaded from [<u> this repo </u>](https://github.com/ZJU-DAILY/ST2Vec).  
The dataset `Porto` can be downloaded from [<u> this repo </u>](https://github.com/chenyuqi990215/RNTrajRec).  
The dataset `Geolife` can be downloaded from [<u> this link </u>](https://www.microsoft.com/en-us/download/details.aspx?id=52367&from=https%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2Fb16d359d-d164-469e-9fd4-daa38f2b2e13%2F).  
The dataset `TaxiBJ` can be downloaded from [<u> this link </u>](https://www.microsoft.com/en-us/download/details.aspx?id=52367&from=https%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2Fb16d359d-d164-469e-9fd4-daa38f2b2e13%2F) and its data format can be found from [<u> this repo </u>](https://github.com/panzheyi/ST-MetaNet).


## Run code
The scripts are shown in `scripts/`.  
E.g., to run experiments on trajectory recovery, the script is
``` 
usage: run_recovery_task.py [--dataset DATASET] [--model_name MODEL] [--num_epochs EPOCHS] [--batch_size SIZE] [--phase PHASE] [--gpu GPU_ID] [--seed SEED] [--num_virtual_tokens TOKEN_NUM] [--num_augment_epochs AUG_EPOCHS]

optional arguements:
--dataset               dataset name
--model_name            downstream model name
--num_epochs            number of epochs to train downstream model (following the original paper)
--batch_size            batch size for model training
--phase                 training / test / augment phase (for init. training, evaluation, and augmentation)
--gpu                   gpu id
--seed                  seed
--num_virtual_tokens    number of virtual tokens
--num_augment_epochs    number of epochs for data augmentation
--mixup                 store_true type (whether to use mixup augmentation) 
```

## Acknowledgement
The code used for downstream tasks is adapted from their original github repos.
