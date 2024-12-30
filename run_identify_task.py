import argparse
import os
import json
import random
import numpy as np
import torch
from torch.optim import AdamW

from utils.data import ModeIdentifyDataset
from utils.train import ModeIdentifyTrainer
from models.identify_models import SECA, CNNSECA
from augment.augment_config import PointUnionConfig, TrajUnionConfig, TaskType
from augment.point_union import PointUnion
from augment.trajectory_union import TrajUnion


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="transportation mode identification")
    parser.add_argument("--dataset", type=str, default="Geolife")
    parser.add_argument("--model_name", type=str, default="SECA")
    parser.add_argument("--num_epochs", type=int, default=20, help="epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--phase", type=str, default=None, help="select from `train`, `test`, `augment`")
    parser.add_argument("--saved_path", type=str, default=None, help="model checkpoint")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--mixup", action="store_true", help="mixup data augmentation")

    parser.add_argument("--augment_type", type=str, default="PointUnion")
    parser.add_argument("--union_ratio", type=float, default=0.05)
    parser.add_argument("--num_virtual_tokens", type=int, default=20)
    parser.add_argument("--num_augment_epochs", type=int, default=20)
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    traj_dir = os.path.join("./data", args.dataset)
    phase = args.phase
    with open("configs/identify_config.json", 'r') as config_file:
        configs = json.load(config_file)
        config = configs[args.model_name]
    input_dim = config["input_dim"]
    ckpt_dir = f"./ckpt/{args.model_name}-{args.dataset}-{input_dim}" if args.saved_path is None else args.saved_path
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    train_dataset = ModeIdentifyDataset(traj_dir, input_dim, "train", args.model_name, maximal_num_points=config["max_length"])
    valid_dataset = ModeIdentifyDataset(traj_dir, input_dim, "valid", args.model_name, maximal_num_points=config["max_length"])
    test_dataset = ModeIdentifyDataset(traj_dir, input_dim, "test", args.model_name, maximal_num_points=config["max_length"])

    model_params = {
        "input_dim": input_dim,
        "num_classes": config["num_classes"],
        "max_length": config["max_length"],
        "hidden_dims": config["hidden_dims"]
    }
    if args.model_name == "SECA":
        model = SECA(**model_params).to(device)
    else:
        model = CNNSECA(**model_params).to(device)

    optim = AdamW(model.parameters(), lr=config["learning_rate"])
    trainer = ModeIdentifyTrainer(model, train_dataset, valid_dataset, optim,
                                  num_epochs=args.num_epochs,
                                  data_collator=ModeIdentifyDataset.collate_fn,
                                  saved_dir=ckpt_dir,
                                  batch_size=args.batch_size,
                                  shuffle=config["shuffle"],
                                  num_workers=8,
                                  pin_memory=True,
                                  device=device,
                                  mixup=args.mixup)
    if (phase is None) or (phase == "train"):
        trainer.train()
        if args.mixup:
            trainer.mixup_train(max_length=config["max_length"] if args.model_name == "SECA" else None)
        metric_result = trainer.evaluate(test_dataset)
        print("ACC: {:.4f}\tRecall: {:.4f}\tPrec: {:.4f}\tMacro F1: {:.4f}\tWeighted F1: {:.4f}\n"
              .format(metric_result[0],
                      metric_result[1],
                      metric_result[2],
                      metric_result[3],
                      metric_result[4]))
    if (phase is None) or (phase == "test"):
        trainer.load_model()
        metric_result = trainer.evaluate(test_dataset)
        print("ACC: {:.4f}\tRecall: {:.4f}\tPrec: {:.4f}\tMacro F1: {:.4f}\tWeighted F1: {:.4f}\n"
              .format(metric_result[0],
                      metric_result[1],
                      metric_result[2],
                      metric_result[3],
                      metric_result[4]))
    if (phase is None) or (phase == "augment"):
        trainer.load_model()
        if args.model_name == "SECA":
            model_kwargs = dict(max_length=config["max_length"])
        else:
            model_kwargs = dict()
        if args.augment_type == "PointUnion":
            augment_config = PointUnionConfig(
                TaskType.TYPE_IDENTIFY,
                model_name=args.model_name,
                virtual_dim=config["input_dim"],
                num_virtual_tokens=args.num_virtual_tokens,
                num_epochs=args.num_augment_epochs
            )
            augmentor = PointUnion(augment_config, trainer, None, device)
            augment_result = augmentor.augment_points(
                test_dataset,
                max_length=config["max_length"]
            )
        elif args.augment_type == "TrajUnion":
            augment_dataset = ModeIdentifyDataset(traj_dir, input_dim, "augment", args.model_name, maximal_num_points=config["max_length"])
            augment_config = TrajUnionConfig(
                task_type=TaskType.TYPE_IDENTIFY,
                model_name=args.model_name,
                num_augments=int(args.union_ratio * len(train_dataset)),
                proj_interval=200,
                save_interval=200
            )
            augmentor = TrajUnion(
                config=augment_config,
                trainer=trainer,
                augment_dataset=augment_dataset,
                device=device
            )
            train_subset = augmentor.select_augmentation(model_kwargs)
            trainer.train_dataset = train_subset
            trainer.saved_dir = os.path.join(ckpt_dir, "traj_union")
            trainer.train()
            augment_result = trainer.evaluate(test_dataset)
        else:
            raise NotImplementedError("Please specify the way of data augmentation.")
        print("ACC: {:.4f}\tRecall: {:.4f}\tPrec: {:.4f}\tMacro F1: {:.4f}\tWeighted F1: {:.4f}\n"
              .format(augment_result[0],
                      augment_result[1],
                      augment_result[2],
                      augment_result[3],
                      augment_result[4]))


if __name__ == "__main__":
    main()
