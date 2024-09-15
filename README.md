# VTDNet

This repository contains the implementation of **VTDNet**, a deep learning model implemented using **PyTorch**. The project includes various utilities for generating synthetic data, training the model, and evaluating its performance using metrics such as PEHE (Precision in Estimation of Heterogeneous Effect) and eATE (expected Average Treatment Effect).

## Requirements

The following software is required to use this repository:

- **Python 3.10.14** 
- **PyTorch 1.13.1**
- **tqdm 4.66.4** 
- **SciPy 1.13.1** 
- **scikit-learn 1.4.2** 

## Installation

To get started with **VTDNet**, follow these steps:

1. **Clone this repository**:

   ```bash
   git clone https://github.com/Throwfox/VTDNet.git
   cd VTDNet
   ```

2. **Create a new conda environment**:

   Use the following command to create a new conda environment with the required dependencies:

   ```bash
   conda create --name VTDNet python=3.10.14 tqdm=4.66.4 scipy=1.13.1 scikit-learn=1.4.2
   ```

3. **Activate the conda environment**:

   After creating the environment, activate it using:

   ```bash
   conda activate VTDNet
   ```
   Install pytorch with gpu based on the cuda version: here cuda =11.6 as an example. It may take a while, and please be patient.
   ```bash
   conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
   ```
## Usage

Once the environment is set up, you can use the following commands to generate data, train the model, and test it.

### 1. Generating Synthetic Data

Use the following command to generate synthetic data for training and evaluation; the params need to be set within the file:

```bash
python ./Data/data_syn.py
```

This script will create the synthetic dataset required for model training.

### 2. Training the Model

You can train the model using the generated dataset with the following command, task = 6 means generation of synthetic with gamma =0.6 :

```bash
python VTD_training.py --dataset=syn --task=6 --cuda=0 --resume=0
```

- `--dataset=syn`: Specifies the dataset to use (synthetic in this case).
- `--task=6`: Specifies the task identifier for training.
- `--cuda=0`: Specifies the CUDA device to use (change this to the appropriate GPU index if needed).
- `--resume=0`: If you want to resume training from a previous checkpoint, set this to `1`. Setting it to `0` starts training from scratch.

### 3. Testing the Model (PEHE / eATE)

Once training is complete, you can evaluate the model using PEHE and eATE metrics. Use the following command to perform inference on the synthetic dataset:

```bash
python VTD_inference.py --task=6
```

- `--task=6`: Specifies the task for which you want to run the inference.

The results will be printed to the console, and relevant metrics such as PEHE and eATE will be computed.

## Additional Options

- **Checkpointing**: The training script automatically saves model checkpoints during training. If you wish to resume training, you can use the `--resume=1` flag when calling the `VTD_training.py` script.
  
- **CUDA Support**: By default, the training and inference scripts use GPU acceleration with CUDA. You can specify which GPU to use by adjusting the `--cuda` flag (e.g., `--cuda=0` to use the first GPU).


## Contribution

If you would like to contribute to the project, please fork the repository, make your changes, and submit a pull request. Bug reports and feature requests are also welcome!

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For any questions or issues, please contact:

- **Hao Dai**
- Email: haodai@ufl.edu
```
