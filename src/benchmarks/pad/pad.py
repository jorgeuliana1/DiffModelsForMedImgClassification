# -*- coding: utf-8 -*-
"""
Original author: André Pacheco
Email: pacheco.comp@gmail.com

Modified by: Jorge Uliana
Email: ulianamjjorge@gmail.com

"""

import logging
import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from aug_pad import ImgEvalTransform, ImgTrainTransform
from preprocess import prepare_data
from raug.eval import test_model
from raug.loader import get_data_loader
from raug.train import fit_model
from raug.utils.loader import get_labels_frequency
from sacred import Experiment
from sacred.observers import FileStorageObserver

# Including the path to the models folder
sys.path.insert(0, os.environ["MY_MODELS_PATH"])
from my_model import set_model  # noqa

# Starting sacred experiment
ex = Experiment()


@ex.config
def cnfg():
    # Preparing data before anything
    prepare_data()

    # Dataset variables
    _folder = 1
    _base_path = os.path.join("/app/datasets", "PAD-UFES-20")

    _csv_path_train = os.path.join(_base_path, "pad-ufes-20_parsed_folders.csv")  # noqa
    _csv_path_test = os.path.join(_base_path, "pad-ufes-20_parsed_test.csv")  # noqa
    _imgs_folder_train = os.path.join(_base_path, "images")  # noqa

    # Loading configs from yaml
    with open("/app/src/config.yaml", "r") as yaml_f:
        cfgs = yaml.safe_load(yaml_f)

    # Dataset variables
    dataset_cfg = cfgs["dataset"]
    _use_meta_data = dataset_cfg["use_meta_data"]  # noqa
    _neurons_reducer_block = dataset_cfg["neurons_reducer_block"]  # noqa
    _comb_method = dataset_cfg["comb_method"]  # noqa
    _comb_config = dataset_cfg["comb_config"]  # noqa

    # Training variables
    training_cfg = cfgs["training"]
    _batch_size = training_cfg["batch_size"]  # noqa
    _epochs = training_cfg["epochs"]  # noqa
    _best_metric = training_cfg["best_metric"]  # noqa
    _pretrained = training_cfg["pretrained"]  # noqa
    _lr_init = training_cfg["lr_init"]  # noqa
    _sched_factor = training_cfg["sched_factor"]  # noqa
    _sched_min_lr = training_cfg["sched_min_lr"]  # noqa
    _sched_patience = training_cfg["sched_patience"]  # noqa
    _early_stop = training_cfg["early_stop"]  # noqa
    _metric_early_stop = training_cfg["metric_early_stop"]  # noqa
    _weights = training_cfg["weights"]  # noqa

    # DataLoader variables
    dataloader_cfg = cfgs["dataloader"]
    _num_workers = dataloader_cfg["num_workers"]  # noqa

    _model_name = "mobilenet"
    _save_dir = f"{_comb_method}_{_model_name}_fold_{_folder}_{str(time.time()).replace('.', '')}"
    _save_folder = os.path.join("/app", "results", _save_dir)

    # This is used to configure the sacred storage observer. In brief, it says to sacred to save its stuffs in
    # _save_folder. You don't need to worry about that.
    SACRED_OBSERVER = FileStorageObserver(_save_folder)
    ex.observers.append(SACRED_OBSERVER)


@ex.automain
def main(
    _folder,
    _csv_path_train,
    _imgs_folder_train,
    _lr_init,
    _sched_factor,
    _sched_min_lr,
    _sched_patience,
    _batch_size,
    _epochs,
    _early_stop,
    _weights,
    _model_name,
    _pretrained,
    _save_folder,
    _csv_path_test,
    _best_metric,
    _neurons_reducer_block,
    _comb_method,
    _comb_config,
    _use_meta_data,
    _metric_early_stop,
    _num_workers,
):

    meta_data_columns = [
        "smoke_False",
        "smoke_True",
        "drink_False",
        "drink_True",
        "background_father_POMERANIA",
        "background_father_GERMANY",
        "background_father_BRAZIL",
        "background_father_NETHERLANDS",
        "background_father_ITALY",
        "background_father_POLAND",
        "background_father_UNK",
        "background_father_PORTUGAL",
        "background_father_BRASIL",
        "background_father_CZECH",
        "background_father_AUSTRIA",
        "background_father_SPAIN",
        "background_father_ISRAEL",
        "background_mother_POMERANIA",
        "background_mother_ITALY",
        "background_mother_GERMANY",
        "background_mother_BRAZIL",
        "background_mother_UNK",
        "background_mother_POLAND",
        "background_mother_NORWAY",
        "background_mother_PORTUGAL",
        "background_mother_NETHERLANDS",
        "background_mother_FRANCE",
        "background_mother_SPAIN",
        "age",
        "pesticide_False",
        "pesticide_True",
        "gender_FEMALE",
        "gender_MALE",
        "skin_cancer_history_True",
        "skin_cancer_history_False",
        "cancer_history_True",
        "cancer_history_False",
        "has_piped_water_True",
        "has_piped_water_False",
        "has_sewage_system_True",
        "has_sewage_system_False",
        "fitspatrick_3.0",
        "fitspatrick_1.0",
        "fitspatrick_2.0",
        "fitspatrick_4.0",
        "fitspatrick_5.0",
        "fitspatrick_6.0",
        "region_ARM",
        "region_NECK",
        "region_FACE",
        "region_HAND",
        "region_FOREARM",
        "region_CHEST",
        "region_NOSE",
        "region_THIGH",
        "region_SCALP",
        "region_EAR",
        "region_BACK",
        "region_FOOT",
        "region_ABDOMEN",
        "region_LIP",
        "diameter_1",
        "diameter_2",
        "itch_False",
        "itch_True",
        "itch_UNK",
        "grew_False",
        "grew_True",
        "grew_UNK",
        "hurt_False",
        "hurt_True",
        "hurt_UNK",
        "changed_False",
        "changed_True",
        "changed_UNK",
        "bleed_False",
        "bleed_True",
        "bleed_UNK",
        "elevation_False",
        "elevation_True",
        "elevation_UNK",
    ]

    _metric_options = {
        "save_all_path": os.path.join(_save_folder, "best_metrics"),
        "pred_name_scores": "predictions_best_test.csv",
        "normalize_conf_matrix": True,
    }
    _checkpoint_best = os.path.join(_save_folder, "best-checkpoint/best-checkpoint.pth")

    # Loading the csv file
    csv_all_folders = pd.read_csv(_csv_path_train)

    print("- Loading validation data...")
    val_csv_folder = csv_all_folders[(csv_all_folders["folder"] == _folder)]
    train_csv_folder = csv_all_folders[csv_all_folders["folder"] != _folder]

    # Loading validation data
    val_imgs_id = val_csv_folder["img_id"].values
    val_imgs_path = [f"{_imgs_folder_train}/{img_id}" for img_id in val_imgs_id]
    val_labels = val_csv_folder["diagnostic_number"].values
    if _use_meta_data:
        val_meta_data = val_csv_folder[meta_data_columns].values
        print(f"-- Using {len(meta_data_columns)} meta-data features")
    else:
        print("-- No metadata")
        val_meta_data = None
    val_data_loader = get_data_loader(
        val_imgs_path,
        val_labels,
        val_meta_data,
        transform=ImgEvalTransform(),
        batch_size=_batch_size,
        shuf=True,
        num_workers=_num_workers,
        pin_memory=True,
    )
    print(
        f"-- Validation partition loaded with {len(val_data_loader) * _batch_size} images"
    )

    print("- Loading training data...")
    train_imgs_id = train_csv_folder["img_id"].values
    train_imgs_path = [f"{_imgs_folder_train}/{img_id}" for img_id in train_imgs_id]
    train_labels = train_csv_folder["diagnostic_number"].values
    if _use_meta_data:
        train_meta_data = train_csv_folder[meta_data_columns].values
        print(f"-- Using {len(meta_data_columns)} meta-data features")
    else:
        print("-- No metadata")
        train_meta_data = None
    train_data_loader = get_data_loader(
        train_imgs_path,
        train_labels,
        train_meta_data,
        transform=ImgTrainTransform(),
        batch_size=_batch_size,
        shuf=True,
        num_workers=_num_workers,
        pin_memory=True,
    )
    print(
        f"-- Training partition loaded with {len(train_data_loader) * _batch_size} images"
    )

    ser_lab_freq = get_labels_frequency(train_csv_folder, "diagnostic", "img_id")
    _labels_name = ser_lab_freq.index.values
    _freq = ser_lab_freq.values

    print(f"- Loading {_model_name}")

    model = set_model(
        _model_name,
        len(_labels_name),
        neurons_reducer_block=_neurons_reducer_block,
        comb_method=_comb_method,
        comb_config=_comb_config,
        pretrained=_pretrained,
    )

    if _weights == "frequency":
        _weights = (_freq.sum() / _freq).round(3)

    loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(_weights).cuda())
    optimizer = optim.SGD(
        model.parameters(), lr=_lr_init, momentum=0.9, weight_decay=0.001
    )
    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=_sched_factor,
        min_lr=_sched_min_lr,
        patience=_sched_patience,
    )

    print("- Starting the training phase...")
    fit_model(
        model,
        train_data_loader,
        val_data_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=_epochs,
        epochs_early_stop=_early_stop,
        save_folder=_save_folder,
        initial_model=None,
        metric_early_stop=_metric_early_stop,
        device=None,
        schedule_lr=scheduler_lr,
        config_bot=None,
        model_name="CNN",
        resume_train=False,
        history_plot=True,
        val_metrics=["balanced_accuracy"],
        best_metric=_best_metric,
    )
    # Testing the validation partition
    print("- Evaluating the validation partition...")
    test_model(
        model,
        val_data_loader,
        checkpoint_path=_checkpoint_best,
        loss_fn=loss_fn,
        save_pred=True,
        partition_name="eval",
        metrics_to_comp="all",
        class_names=_labels_name,
        metrics_options=_metric_options,
        apply_softmax=True,
        verbose=False,
    )

    print("- Loading test data...")
    csv_test = pd.read_csv(_csv_path_test)
    test_imgs_id = csv_test["img_id"].values
    test_imgs_path = [f"{_imgs_folder_train}/{img_id}" for img_id in test_imgs_id]
    test_labels = csv_test["diagnostic_number"].values
    if _use_meta_data:
        test_meta_data = csv_test[meta_data_columns].values
        print(f"-- Using {len(meta_data_columns)} meta-data features")
    else:
        test_meta_data = None
        print("-- No metadata")

    _metric_options = {
        "save_all_path": os.path.join(_save_folder, "test_pred"),
        "pred_name_scores": "predictions.csv",
        "normalize_conf_matrix": True,
    }
    test_data_loader = get_data_loader(
        test_imgs_path,
        test_labels,
        test_meta_data,
        transform=ImgEvalTransform(),
        batch_size=_batch_size,
        shuf=False,
        num_workers=_num_workers,
        pin_memory=True,
    )

    # Testing the test partition
    print("\n- Evaluating the validation partition...")
    test_model(
        model,
        test_data_loader,
        checkpoint_path=None,
        metrics_to_comp="all",
        class_names=_labels_name,
        metrics_options=_metric_options,
        save_pred=True,
        verbose=False,
    )
