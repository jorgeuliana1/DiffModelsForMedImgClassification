# -*- coding: utf-8 -*-
"""
Author: Jorge Uliana
Email: ulianamjjorge@gmail.com

Script for train and evaluation on P-NDB-UFES dataset
"""

import json
import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from aug_ndb import ImgEvalTransform, ImgTrainTransform
from preprocess import prepare_data
from raug.eval import test_model
from raug.loader import get_data_loader
from raug.train import fit_model
from raug.utils.loader import get_labels_frequency
from sacred import Experiment
from sacred.observers import FileStorageObserver

# Including the path to the models folder
sys.path.insert(0, os.environ["MY_MODELS_PATH"])
from my_model import get_norm_and_size, set_model  # noqa

# Starting sacred experiment
ex = Experiment()

@ex.config
def cnfg():
    # Preparing data before anything
    prepare_data()
    
    # Loading tasks infos from json
    with open("/app/src/benchmarks/ndb/tasks.json", "r") as json_f:
        tasks = json.load(json_f)  # noqa

    # Loading configs from yaml
    with open("/app/src/config.yaml", "r") as yaml_f:
        cfgs = yaml.safe_load(yaml_f)

    task_label = cfgs["ndb"]["task"]
    base_path = os.path.join("/app", "datasets", "NDB-UFES")

    folder = 5
    csv_path_train = os.path.join(  # noqa
        base_path, f"ndbufes_{task_label}_parsed_folders.csv"
    )
    csv_path_test = os.path.join(  # noqa
        base_path, f"ndbufes_{task_label}_parsed_test.csv"
    )
    imgs_folder_train = os.path.join(base_path, "images")  # noqa

    # Dataset variables
    dataset_cfg = cfgs["dataset"]
    use_meta_data = dataset_cfg["use_meta_data"]  # noqa
    neurons_reducer_block = dataset_cfg["neurons_reducer_block"]  # noqa
    comb_method = dataset_cfg["comb_method"]  # noqa
    comb_config = dataset_cfg["comb_config"]  # noqa

    # Training variables
    training_cfg = cfgs["training"]
    batch_size = training_cfg["batch_size"]  # noqa
    epochs = training_cfg["epochs"]  # noqa
    best_metric = training_cfg["best_metric"]  # noqa
    pretrained = training_cfg["pretrained"]  # noqa

    # Keep lr x batch_size proportion. Batch size 30 was the original one.
    # Read more in https://arxiv.org/abs/2006.09092
    # For adaptive optimizers
    # prop = np.sqrt(_batch_size/30.) if _keep_lr_prop else 1
    # For SGD
    keep_lr_prop = training_cfg["keep_lr_prop"]
    prop = batch_size / 30 if keep_lr_prop else 1
    lr_init = training_cfg["lr_init"] * prop  # noqa
    sched_factor = training_cfg["sched_factor"] * prop  # noqa
    sched_min_lr = training_cfg["sched_min_lr"] * prop  # noqa

    sched_patience = training_cfg["sched_patience"]  # noqa
    early_stop = training_cfg["early_stop"]  # noqa
    metric_early_stop = training_cfg["metric_early_stop"]  # noqa
    weights = training_cfg["weights"]  # noqa

    model_name = "mobilenet"
    save_dir = (
        f"{comb_method}_{model_name}_fold_{folder}_{str(time.time()).replace('.', '')}"
    )
    save_folder = os.path.join("/app", "results", save_dir)

    # This is used to configure the sacred storage observer. In brief, it says to sacred to save its stuffs in
    # save_folder. You don't need to worry about that.
    SACRED_OBSERVER = FileStorageObserver(save_folder)
    ex.observers.append(SACRED_OBSERVER)


@ex.automain
def main(
    folder,
    base_path,
    csv_path_train,
    imgs_folder_train,
    lr_init,
    sched_factor,
    sched_min_lr,
    sched_patience,
    batch_size,
    epochs,
    early_stop,
    weights,
    model_name,
    pretrained,
    save_folder,
    csv_path_test,
    best_metric,
    neurons_reducer_block,
    comb_method,
    comb_config,
    use_meta_data,
    metric_early_stop,
    task_label,
    tasks,
):
    meta_data_columns = tasks[task_label]["features"]
    label_name = tasks[task_label]["label"]
    img_path_col = tasks[task_label]["img_col"]

    csv_path_train = os.path.join(base_path, f"ndbufes_{task_label}_parsed_folders.csv")
    csv_path_test = os.path.join(base_path, f"ndbufes_{task_label}_parsed_test.csv")
    imgs_folder_train = os.path.join(base_path, tasks[task_label]["img_path"])

    metric_options = {
        "save_all_path": os.path.join(save_folder, "best_metrics"),
        "pred_name_scores": "predictions_best_test.csv",
        "normalize_conf_matrix": True,
    }
    checkpoint_best = os.path.join(save_folder, "best-checkpoint/best-checkpoint.pth")

    # Loading the csv file
    csv_all_folders = pd.read_csv(csv_path_train)

    print("- Loading validation data...")
    if "synthetic" in csv_all_folders.columns:
        synthetics = csv_all_folders["synthetic"]
    else:
        synthetics = False

    val_csv_folder = csv_all_folders[(csv_all_folders["folder"] == folder) & ~synthetics]
    train_csv_folder = csv_all_folders[csv_all_folders["folder"] != folder]

    transform_param = get_norm_and_size(model_name)

    # Loading validation data
    val_imgs_id = val_csv_folder[img_path_col].values
    val_imgs_path = ["{}/{}".format(imgs_folder_train, img_id) for img_id in val_imgs_id]
    val_labels = val_csv_folder["label_number"].values
    if use_meta_data:
        val_meta_data = val_csv_folder[meta_data_columns].values
        print("-- Using {} meta-data features".format(len(meta_data_columns)))
    else:
        print("-- No metadata")

    val_meta_data = None
    val_data_loader = get_data_loader(
        val_imgs_path,
        val_labels,
        val_meta_data,
        transform=ImgEvalTransform(*transform_param),
        batch_size=batch_size,
        shuf=True,
        num_workers=2,
        pin_memory=True,
    )
    print(
        "-- Validation partition loaded with {} images".format(
            len(val_data_loader) * batch_size
        )
    )

    print("- Loading training data...")
    train_imgs_id = train_csv_folder[img_path_col].values
    train_imgs_path = [
        "{}/{}".format(imgs_folder_train, img_id) for img_id in train_imgs_id
    ]
    train_labels = train_csv_folder["label_number"].values
    if use_meta_data:
        train_meta_data = train_csv_folder[meta_data_columns].values
        print("-- Using {} meta-data features".format(len(meta_data_columns)))
    else:
        print("-- No metadata")
    train_meta_data = None
    train_data_loader = get_data_loader(
        train_imgs_path,
        train_labels,
        train_meta_data,
        transform=ImgTrainTransform(*transform_param),
        batch_size=batch_size,
        shuf=True,
        num_workers=2,
        pin_memory=True,
    )
    print(
        "-- Training partition loaded with {} images".format(
            len(train_data_loader) * batch_size
        )
    )

    ser_lab_freq = get_labels_frequency(train_csv_folder, label_name, img_path_col)
    labels_name = ser_lab_freq.index.values
    freq = ser_lab_freq.values
    
    print("- Loading", model_name)

    model = set_model(
        model_name,
        len(labels_name),
        neurons_reducer_block=neurons_reducer_block,
        comb_method=comb_method,
        comb_config=comb_config,
        pretrained=pretrained,
    )
  
    if weights == "frequency":
        weights = (freq.sum() / freq).round(3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(weights).to(device))
    optimizer = optim.SGD(
        model.parameters(), lr=lr_init, momentum=0.9, weight_decay=0.001
    )
    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=sched_factor, min_lr=sched_min_lr, patience=sched_patience
    )

    print("- Starting the training phase...")
    print("-" * 50)
    fit_model(
        model,
        train_data_loader,
        val_data_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        epochs_early_stop=early_stop,
        save_folder=save_folder,
        initial_model=None,
        metric_early_stop=metric_early_stop,
        device=None,
        schedule_lr=scheduler_lr,
        config_bot=None,
        model_name="CNN",
        resume_train=False,
        history_plot=True,
        val_metrics=["balanced_accuracy"],
        best_metric=best_metric,
    )

    # Testing the validation partition
    print("- Evaluating the validation partition...")
    test_model(
        model,
        val_data_loader,
        checkpoint_path=checkpoint_best,
        loss_fn=loss_fn,
        save_pred=True,
        partition_name="eval",
        metrics_to_comp="all",
        class_names=labels_name,
        metrics_options=metric_options,
        apply_softmax=True,
        verbose=False,
    )

    print("- Loading test data...")
    csv_test = pd.read_csv(csv_path_test)
    test_imgs_id = csv_test[img_path_col].values
    test_imgs_path = [
        "{}/{}".format(imgs_folder_train, img_id) for img_id in test_imgs_id
    ]
    test_labels = csv_test["label_number"].values
    if use_meta_data:
        test_meta_data = csv_test[meta_data_columns].values
        print("-- Using {} meta-data features".format(len(meta_data_columns)))
    else:
        test_meta_data = None
        print("-- No metadata")

    metric_options = {
        "save_all_path": os.path.join(save_folder, "test_pred"),
        "pred_name_scores": "predictions.csv",
        "normalize_conf_matrix": True,
    }

    test_data_loader = get_data_loader(
        test_imgs_path,
        test_labels,
        test_meta_data,
        transform=ImgEvalTransform(*transform_param),
        batch_size=batch_size,
        shuf=False,
        num_workers=2,
        pin_memory=True,
    )
    
    # Testing the test partition
    print("\n- Evaluating the validation partition...")
    test_model(
        model,
        test_data_loader,
        checkpoint_path=checkpoint_best,
        metrics_to_comp="all",
        class_names=labels_name,
        metrics_options=metric_options,
        save_pred=True,
        verbose=False,
    )
