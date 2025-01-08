# CV Master Project: Facial Color Clothes Classification

## Getting Started

### Preparing the Dataset
To set up the dataset for this project, refer to the detailed instructions in [`/data/README.md`](./data/README.md).

### Training the Model
1. Update the paths for the dataset and checkpoint in `train.py`:
   - Locate the variables for dataset and checkpoint paths in `train.py`.
   - Modify them to match your local file structure.

2. Train the model using the following command:
   ```bash
   python3 train.py --arch {model_name}
   ```

## Benchmark Results

| Model        | Data                            | Accuracy (%) | Inference speed           | Params (M)                  | Note                      |
|--------------|---------------------------------|--------------|---------------------------|---------------------------  |---------------------------|
| MobileNetV3  | Resize(256), Crop(224)          | 81.92        |---------------------------|---------------------------  |---------------------------|
| MobileNetV3  | Resize(224)                     | @@@@         |---------------------------|---------------------------  |---------------------------|
| MobileNetV2  | Resize(256), Crop(224)          | 81.34        |---------------------------|---------------------------  |---------------------------|
| MobileNetV2  | Resize(224)                     | @@@@         |---------------------------|---------------------------  |---------------------------|
| ResNet50     | Resize(256), Crop(224)          | 78.09        |---------------------------|---------------------------  |---------------------------|
| ResNet50     | Resize(224)                     | @@@@         |---------------------------|---------------------------  |---------------------------|
| Alexnet      | Resize(256), Crop(224)          | 79.46        |---------------------------|---------------------------  |---------------------------|
| Alexnet      | Resize(224)                     | 78.98        |---------------------------|---------------------------  |---------------------------|

