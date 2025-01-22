![Static Badge](https://img.shields.io/badge/Classification-FF0000)
![Static Badge](https://img.shields.io/badge/Neuroimaging-FF0000)
![Static Badge](https://img.shields.io/badge/AI-8A2BE2)
![Static Badge](https://img.shields.io/badge/Python-8A2BE2)
![Static Badge](https://img.shields.io/badge/PET%20/%20MRI-4CAF50)

# AI-driven Classification of Alzheimer's Disease and Mild Cognitive Impairment
## Overview

This project employs advanced machine learning algorithms to classify Alzheimer's Disease with high accuracy. By leveraging a dataset of cognitive and biological markers, the code achieves remarkable performance metrics, including:
(Accurate measurements are not available until the paper is published!)

- **Accuracy**: >80%
- **Sensitivity**: >80%
- **Specificity**: >80%
- **F1 Score**: >80%

The implementation is designed for ease of use and reproducibility, enabling researchers and practitioners to effectively diagnose Alzheimer's Disease.

![Image](https://github.com/user-attachments/assets/cf6cb9ad-0d65-4713-aacc-a9d37d9d88ba)

## Key Features

- Implements a Multinomial Logistic Regression (MLR) & Multi-Layer Perceptron (MLP)
- Multi-class classification.
- Optimized hyperparameters for high performance.
- Outputs detailed evaluation metrics (accuracy, sensitivity, specificity, F1 score).
- Configurable for different datasets.

## Data

### Data Source

This project uses data from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/). ADNI is a longitudinal multicenter study designed to develop clinical, imaging, genetic, and biochemical biomarkers for the early detection and tracking of Alzheimer's disease.

To access ADNI data, you must apply for access through their [Data Access Application](https://adni.loni.usc.edu/data-samples/access-data/).

Please note that this repository does not contain any ADNI data due to restrictions on data use agreements.

### Our Dataset

We collected a dataset consisting of MRI T1-W, FDG-PET, comorbidities, and demographic information for all subjects from the ADNI database.
The dataset template template has been added to the repository as Features-template.xlsx

## Data Analysis

The PET and MRI images were preprocessed using Python and SPM12.
Data cleaning, outlier handling, and data manipulation were implemented using Python.
MLR and MLP models were developed as follows.

![Image](https://github.com/user-attachments/assets/b22d43c2-aee0-4a3d-b3a4-0fe11246574d)

# Model Summaries

## MLR Summary

| Layer (type)                     | Input Shape     | Output Shape   | Param # | Trainable |
|----------------------------------|-----------------|----------------|---------|-----------|
| LogisticRegression               | [8, 1, 237]    | [8, 1, 3]      | --      | True      |
| ├─ Linear (linear)               | [8, 1, 237]    | [8, 1, 3]      | 714     | True      |

**Total Parameters:** 714  
**Trainable Parameters:** 714  
**Non-trainable Parameters:** 0  
**Total Mult-Adds (M):** 0.01  

| **Resource**                | **Size (MB)** |
|-----------------------------|---------------|
| Input size                  | 0.01          |
| Forward/backward pass size  | 0.00          |
| Params size                 | 0.00          |
| **Estimated Total Size**    | **0.01**      |

---

## MLP Summary

| Layer (type)                     | Input Shape     | Output Shape   | Param # | Trainable |
|----------------------------------|-----------------|----------------|---------|-----------|
| MLP                              | [8, 1, 237]    | [8, 1, 3]      | --      | True      |
| ├─ Linear (fc1)                  | [8, 1, 237]    | [8, 1, 64]     | 15,232  | True      |
| ├─ Linear (fc2)                  | [8, 1, 64]     | [8, 1, 64]     | 4,160   | True      |
| ├─ Linear (fc3)                  | [8, 1, 64]     | [8, 1, 32]     | 2,080   | True      |
| ├─ Linear (fc4)                  | [8, 1, 32]     | [8, 1, 32]     | 1,056   | True      |
| ├─ Linear (fc5)                  | [8, 1, 32]     | [8, 1, 3]      | 99      | True      |

**Total Parameters:** 22,627  
**Trainable Parameters:** 22,627  
**Non-trainable Parameters:** 0  
**Total Mult-Adds (M):** 0.18  

| **Resource**                | **Size (MB)** |
|-----------------------------|---------------|
| Input size                  | 0.01          |
| Forward/backward pass size  | 0.01          |
| Params size                 | 0.09          |
| **Estimated Total Size**    | **0.11**      |

---

This structure is neat, accessible, and easy to edit. Let me know if you’d like to make additional adjustments!


## Requirements

To run this project, install the following dependencies:

- Python 3.8+
- Required libraries (install via `requirements.txt`):
  ```bash
  pip install -r requirements.txt
  ```

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

For questions or collaborations, contact: [GitHub Profile](https://github.com/taha-parsayan)

