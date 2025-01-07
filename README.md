![Static Badge](https://img.shields.io/badge/Classification-FF0000)
![Static Badge](https://img.shields.io/badge/Neuroimaging-FF0000)
![Static Badge](https://img.shields.io/badge/AI-8A2BE2)
![Static Badge](https://img.shields.io/badge/Python-8A2BE2)
![Static Badge](https://img.shields.io/badge/PET%20/%20MRI-4CAF50)

# AI-driven Classification of Alzheimer's Disease and Mild Cognitive Impairment
## Overview

This project employs advanced machine learning algorithms to classify Alzheimer's Disease with high accuracy. By leveraging a dataset of cognitive and biological markers, the code achieves remarkable performance metrics, including:

- **Accuracy**: >80%
- **Sensitivity**: >80%
- **Specificity**: >80%
- **F1 Score**: >80%

The implementation is designed for ease of use and reproducibility, enabling researchers and practitioners to effectively diagnose Alzheimer's Disease.

## Key Features

- Implements a Multinomial Logistic Regression (MLR) & Multi-Layer Perceptron (MLP)
- Multi-class classification.
- Optimized hyperparameters for high performance.
- Outputs detailed evaluation metrics (accuracy, sensitivity, specificity, F1 score).
- Configurable for different datasets.

## MLR Summary
========================================================================================================================
Layer (type (var_name))                  Input Shape          Output Shape         Param #              Trainable
========================================================================================================================
LogisticRegression (LogisticRegression)  [8, 1, 237]          [8, 1, 3]            --                   True
├─Linear (linear)                        [8, 1, 237]          [8, 1, 3]            714                  True
========================================================================================================================
Total params: 714
Trainable params: 714
Non-trainable params: 0
Total mult-adds (M): 0.01
========================================================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.01
========================================================================================================================


## MLP Summary
========================================================================================================================
Layer (type (var_name))                  Input Shape          Output Shape         Param #              Trainable
========================================================================================================================
MLP (MLP)                                [8, 1, 237]          [8, 1, 3]            --                   True
├─Linear (fc1)                           [8, 1, 237]          [8, 1, 64]           15,232               True
├─Linear (fc2)                           [8, 1, 64]           [8, 1, 64]           4,160                True
├─Linear (fc3)                           [8, 1, 64]           [8, 1, 32]           2,080                True
├─Linear (fc4)                           [8, 1, 32]           [8, 1, 32]           1,056                True
├─Linear (fc5)                           [8, 1, 32]           [8, 1, 3]            99                   True
========================================================================================================================
Total params: 22,627
Trainable params: 22,627
Non-trainable params: 0
Total mult-adds (M): 0.18
========================================================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 0.01
Params size (MB): 0.09
Estimated Total Size (MB): 0.11
========================================================================================================================

## Requirements

To run this project, install the following dependencies:

- Python 3.8+
- Required libraries (install via `requirements.txt`):
  ```bash
  pip install -r requirements.txt
  ```

## Dataset

The dataset used should include relevant features for Alzheimer's Disease classification. Ensure your dataset is in CSV format with appropriate preprocessing applied. The dataset template template has been added to the repository as Features-template.xlsx

## Usage

### Step 1: Clone the Repository

```bash
git clone https://github.com/taha-parsayan/Classification-of-Alzheimers-Disease-using-AI-algorithms.git
cd Classification-of-Alzheimers-Disease-using-AI-algorithms
```

### Step 2: Prepare the Data

Place your dataset in the `data/` directory. Modify the data loading section in `main.py` to specify your dataset path and adjust preprocessing steps as needed.

### Step 3: Run the Code

Execute the script to train and evaluate the model:

```bash
python main.py
```

### Step 4: View Results

The results, including accuracy, sensitivity, specificity, and F1 score, will be displayed in the console and saved to `results/`.

## Code Structure

```
Classification-of-Alzheimers-Disease-using-AI-algorithms/
│
├── main.py                # Main script for training and evaluation
├── data/                  # Directory for the dataset
├── models/                # Directory for saving trained models
├── results/               # Directory for saving results
├── utils.py               # Utility functions (if applicable)
└── requirements.txt       # Required Python libraries
```

## Performance Metrics

| Metric          | Value (%) |
|------------------|-----------|
| **Accuracy**     | >80       |
| **Sensitivity**  | >80       |
| **Specificity**  | >80       |
| **F1 Score**     | >80       |

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

**Taha Parsayan**  
Ph.D. Student in AI  
For questions or collaborations, contact: [GitHub Profile](https://github.com/taha-parsayan)

---

Let me analyze the `main.py` file to ensure this README reflects its functionality accurately. One moment!

The `main.py` file appears to implement a deep learning pipeline for classifying Alzheimer's Disease. The initial inspection shows that it uses PyTorch for model implementation, along with several libraries for data preprocessing, statistical analysis, and evaluation metrics.

Here’s how the README should be adjusted based on these observations:

---

# Classification of Alzheimer's Disease Using AI Algorithms

## Overview

This project utilizes a PyTorch-based Multi-Layer Perceptron (MLP) to classify Alzheimer's Disease. The pipeline includes data preprocessing, statistical analysis, and robust evaluation metrics to ensure the model's effectiveness. With an emphasis on reproducibility and performance, this implementation achieves:

- **Accuracy**: >80%
- **Sensitivity**: >80%
- **Specificity**: >80%
- **F1 Score**: >80%

## Key Features

- PyTorch implementation of MLP for multi-class classification.
- Preprocessing and feature selection included.
- Statistical validation using libraries such as `scipy` and `pingouin`.
- Automated evaluation with comprehensive metrics.

---

Would you like me to analyze the entire file in detail to further refine this README?
