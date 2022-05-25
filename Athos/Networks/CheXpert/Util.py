"""

Authors: Saksham Gupta.

Copyright:
Copyright (c) 2020 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

**
Part of code from https://github.com/kamenbliznashki/chexpert
Modified for our purposes.
**

"""


import os
import json
import math
import pickle
import numpy, sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "TFCompiler"))
import DumpTFMtData


class ChexpertSmall(Dataset):
    url = "http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip"
    dir_name = os.path.splitext(os.path.basename(url))[
        0
    ]  # folder to match the filename
    attr_all_names = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]
    # select only the competition labels
    attr_names = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
    ]

    def __init__(self, root, mode="train", transform=None, data_filter=None):
        self.root = root
        self.transform = transform
        assert mode in ["train", "valid"]
        self.mode = mode

        # if mode is train/valid; root is path to data folder with `train`/`valid` csv file to construct dataset.

        self._maybe_process(data_filter)

        data_file = os.path.join(
            self.root, self.dir_name, "valid.pt" if mode in ["valid"] else "train.pt"
        )
        self.data = torch.load(data_file)

        # store index of the selected attributes in the columns of the data for faster indexing
        self.attr_idxs = [self.data.columns.tolist().index(a) for a in self.attr_names]

    def __getitem__(self, idx):
        # 1. select and load image
        img_path = self.data.iloc[idx, 0]  # 'Path' column is 0
        img = Image.open(os.path.join(self.root, img_path)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # 2. select attributes as targets
        attr = self.data.iloc[idx, self.attr_idxs].values.astype(np.float32)
        attr = torch.from_numpy(attr)

        # 3. save index for extracting the patient_id in prediction/eval results as 'CheXpert-v1.0-small/valid/patient64541/study1'
        #    performed using the extract_patient_ids function
        idx = self.data.index[
            idx
        ]  # idx is based on len(self.data); if we are taking a subset of the data, idx will be relative to len(subset);
        # self.data.index(idx) pulls the index in the original dataframe and not the subset

        return img, attr, idx

    def __len__(self):
        return len(self.data)

    def _maybe_process(self, data_filter):
        # Dataset labels are: blank for unmentioned, 0 for negative, -1 for uncertain, and 1 for positive.
        # Process by:
        #    1. fill NAs (blanks for unmentioned) as 0 (negatives)
        #    2. fill -1 as 1 (U-Ones method described in paper)  # TODO -- setup options for uncertain labels
        #    3. apply attr filters as a dictionary {data_attribute: value_to_keep} e.g. {'Frontal/Lateral': 'Frontal'}

        # check for processed .pt files
        train_file = os.path.join(self.root, self.dir_name, "train.pt")
        valid_file = os.path.join(self.root, self.dir_name, "valid.pt")
        if not (os.path.exists(train_file) and os.path.exists(valid_file)):
            # load data and preprocess training data
            valid_df = pd.read_csv(
                os.path.join(self.root, self.dir_name, "valid.csv"),
                keep_default_na=True,
            )
            train_df = self._load_and_preprocess_training_data(
                os.path.join(self.root, self.dir_name, "train.csv"), data_filter
            )

            # save
            torch.save(train_df, train_file)
            torch.save(valid_df, valid_file)

    def _load_and_preprocess_training_data(self, csv_path, data_filter):
        train_df = pd.read_csv(csv_path, keep_default_na=True)

        # 1. fill NAs (blanks for unmentioned) as 0 (negatives)
        # attr columns ['No Finding', ..., 'Support Devices']; note AP/PA remains with NAs for Lateral pictures
        train_df[self.attr_names] = train_df[self.attr_names].fillna(0)

        # 2. fill -1 as 1 (U-Ones method described in paper)  # TODO -- setup options for uncertain labels
        train_df[self.attr_names] = train_df[self.attr_names].replace(-1, 1)

        if data_filter is not None:
            # 3. apply attr filters
            # only keep data matching the attribute e.g. df['Frontal/Lateral']=='Frontal'
            for k, v in data_filter.items():
                train_df = train_df[train_df[k] == v]

            with open(
                os.path.join(
                    os.path.dirname(csv_path), "processed_training_data_filters.json"
                ),
                "w",
            ) as f:
                json.dump(data_filter, f)

        return train_df


def compute_mean_and_std(dataset):
    m = 0
    s = 0
    k = 1
    for img, _, _ in tqdm(dataset):
        x = img.mean().item()
        new_m = m + (x - m) / k
        s += (x - m) * (x - new_m)
        m = new_m
        k += 1
    print("Number of datapoints: ", k)
    return m, math.sqrt(s / (k - 1))


def save_data_as_pickle(dataset, mode, scalingFac):
    preProcessedImgSaveFolder = "./Data_batch"
    filename = os.path.join(
        preProcessedImgSaveFolder, "preprocess_" + mode + "_batch" + ".p"
    )
    features = []
    labels = []
    ids = []
    for img, attr, id in dataset:
        img[...] = img * (1 << scalingFac)
        print("Processed img {}".format(id))
        # print(type(img))
        img = img.reshape(-1, 1)
        # print(img.shape)
        features.append(img)
        labels.append(attr)
        ids.append(id)
    pickle.dump((features, labels, ids), open(filename, "wb"))


def load_preprocess_validation_data(
    preProcessedImgSaveFolder="./Data_batch",
):
    valid_features, valid_labels, valid_ids = pickle.load(
        open(
            os.path.join(preProcessedImgSaveFolder, "preprocess_valid_batch.p"),
            mode="rb",
        )
    )
    return valid_features, valid_labels, valid_ids


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("scale", type=str, help="Scaling Factor.")
    args = parser.parse_args()
    scalingFac = int(args.scale)
    mode = "valid"
    ds = ChexpertSmall(
        "../../HelperScripts/CheXpert",
        mode,
        transform=T.Compose(
            [
                # T.Grayscale(num_output_channels=3),
                T.CenterCrop(320),
                T.ToTensor(),
                T.Normalize(mean=[0.5306], std=[0.0333]),
                T.Lambda(lambda x: torch.flatten(x)),
            ]
        ),
    )
    print("length: ", len(ds))
    print("attributes: ", ds.attr_names)
    # m, s = compute_mean_and_std(ds)
    # print("Dataset mean: {}; dataset std {}".format(m, s))
    save_data_as_pickle(ds, mode, scalingFac)
    print("\n" * 4)

    print("*" * 20)
    id = 1
    print("Sample Image {} from Valid Dataset".format(id))
    print("*" * 20)
    features, labels, ids = load_preprocess_validation_data()
    print(features[id].shape)
    print(features[id])
    print(labels[id])
    print(ids[id])


if __name__ == "__main__":
    main()
