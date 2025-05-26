import os


#################################################################################
#                         run trajectory recovery task                          #
#################################################################################
range_virtual_tokens = [5, 10, 15, 20, 25]
models = ["RNTrajRec", "MTrajRec"]
dataset = "Porto"
for model_name in models:
    for num_tokens in range_virtual_tokens:
        os.system(
            f"python run_recovery_task.py --dataset {dataset} --model_name {model_name} "
            f"--num_epochs 30 --batch_size 64 --phase augment --gpu 0 --seed 2024 "
            f"--num_virtual_tokens {num_tokens} --num_augment_epochs 20"
        )

#################################################################################
#                        run trajectory similarity task                         #
#################################################################################
range_virtual_tokens = [5, 10, 15, 20, 25]
models = ["ST2Vec", "GTS"]
dataset = "tdrive"
for model_name in models:
    for num_tokens in range_virtual_tokens:
        os.system(
            f"python run_similarity_task.py --dataset {dataset} --model_name {model_name} "
            f"--num_epochs 150 --batch_size 64 --phase augment --gpu 0 --seed 2024 "
            f"--num_virtual_tokens {num_tokens} --num_augment_epochs 20"
        )
