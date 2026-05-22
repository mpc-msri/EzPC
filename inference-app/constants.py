from pathlib import Path
from preprocess import get_arr_from_image

# This repository's directory
REPO_DIR = Path(__file__).parent

INPUT_EXAMPLES_DIR = REPO_DIR / "Assets" / "examples"
EXAMPLES = [str(image) for image in INPUT_EXAMPLES_DIR.glob("**/*")]


# ********** USER_INPUTS STARTS **********

# Description
desc = "In this example app, we demonstrate how infer any Chest Xray with a model trained on Chexpert Dataset in a secure manner using EzPC."

# preprocess is a function that takes in an image and returns a numpy array
preprocess = get_arr_from_image

# The input shape of the model, batch size should be 1
Input_Shape = (1, 3, 320, 320)
assert Input_Shape[0] == 1, "Batch size should be 1"
dims = {
    "c": 3,
    "h": 320,
    "w": 320,
}

scale = 15

# Labels
labels_map = {
    0: "No Finding",
    1: "Enlarged Cardiomediastinum",
    2: "Cardiomegaly",
    3: "Lung Lesion",
    4: "Lung Opacity",
    5: "Edema",
    6: "Consolidation",
    7: "Pneumonia",
    8: "Atelectasis",
    9: "Pneumothorax",
    10: "Pleural Effusion",
    11: "Pleural Other",
    12: "Fracture",
    13: "Support Devices",
}

# ********** USER_INPUTS ENDS **********

if dims["c"] == 3:
    mode = "RGB"
elif dims["c"] == 1:
    mode = "L"
