# -*- coding: utf-8 -*-
"""
Original author: André Pacheco
Email: pacheco.comp@gmail.com

Modified by: Jorge Uliana
Email: ulianamjjorge@gmail.com

Script to prepare data to train, validate and test PAD-UFES-20 dataset
"""

import os

import pandas as pd
from raug.utils.loader import label_categorical_to_number, split_k_folder_csv


def prepare_data():
    dataset_name = "PAD-UFES-20"
    dataset_path = os.path.join("/app", "datasets", dataset_name)

    # Verifying for existance of already preprocessed data
    parsed_test_path = os.path.join(dataset_path, "pad-ufes-20_parsed_test.csv")
    parsed_folders_path = os.path.join(dataset_path, "pad-ufes-20_parsed_folders.csv")
    if os.path.exists(parsed_test_path) and os.path.exists(parsed_folders_path):
        return  # There's no need for preprocessing, it's already been performed

    raw_data_csv_path = os.path.join(dataset_path, "metadata.csv")
    raw_data_csv = pd.read_csv(raw_data_csv_path).fillna("EMPTY")

    # Treating columns in order to make it more adequate for NN training/validation
    # untreated_columns refers to the columns which don't require any sort of treatment
    untreated_columns = {
        "img_id",
        "diagnostic",
        "patient_id",
        "lesion_id",
        "biopsed",
    }
    dataset_columns = set(raw_data_csv.columns)
    pretreated_columns = list(dataset_columns - untreated_columns)
    treated_columns = ["age", "diameter_1", "diameter_2"]
    expanded_columns = [
        [f"{column}_{u}" for u in raw_data_csv[column].unique() if u != "EMPTY"]
        for column in pretreated_columns
        if column not in treated_columns
    ]

    for expanded_column in expanded_columns:
        treated_columns += expanded_column

    # Creating a new dataframe for the parsed data:
    new_df = {column: [] for column in treated_columns}

    for _, row in raw_data_csv.iterrows():
        new_col_true = list()
        new_col_false = list()
        for pretreated_column in pretreated_columns:
            row_value = row[pretreated_column]

            # EMPTY fields wont be considered
            if row_value == "EMPTY":
                continue

            # Data in "treated columns" don't need to be treated either
            if pretreated_column in treated_columns:
                new_col_false.append(pretreated_column)
                new_df[pretreated_column].append(row_value)
                continue

            else:
                new_col_true.append(f"{pretreated_column}_{row_value}")

        for x in new_df:
            if x in new_col_true:
                new_df[x].append(1)
            elif x not in new_col_false:
                new_df[x].append(0)

    data_csv = pd.DataFrame.from_dict(new_df)

    # Adding untreated columns:
    for untreated_column in untreated_columns:
        data_csv[untreated_column] = raw_data_csv[untreated_column]

    data = split_k_folder_csv(
        data_csv, "diagnostic", save_path=None, k_folder=6, seed_number=8
    )

    data_test = data[data["folder"] == 6]
    data_train = data[data["folder"] != 6]
    data_test.to_csv(parsed_test_path, index=False)

    label_categorical_to_number(
        parsed_test_path,
        "diagnostic",
        col_target_number="diagnostic_number",
        save_path=parsed_test_path,
    )

    data_train = data_train.reset_index(drop=True)
    data_train = split_k_folder_csv(
        data_train, "diagnostic", save_path=None, k_folder=5, seed_number=8
    )

    label_categorical_to_number(
        data_train,
        "diagnostic",
        col_target_number="diagnostic_number",
        save_path=parsed_folders_path,
    )
