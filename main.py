import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
from scipy.stats import levene
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import ConcatDataset,SubsetRandomSampler
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from sklearn.preprocessing import label_binarize
import joblib
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

torch.manual_seed(27)

#%%

# Import features from the excel file
path = 'Features-2.xlsx'
AD_image_features = pd.read_excel(path, sheet_name='AD')
AD_SUVR_ref = pd.read_excel(path, sheet_name='AD-ref')
AD_medical_features = pd.read_excel(path, sheet_name='AD-medical')

MCI_image_features = pd.read_excel(path, sheet_name='MCI')
MCI_SUVR_ref = pd.read_excel(path, sheet_name='MCI-ref')
MCI_medical_features = pd.read_excel(path, sheet_name='MCI-medical')

NC_image_features = pd.read_excel(path, sheet_name='NC')
NC_SUVR_ref = pd.read_excel(path, sheet_name='NC-ref')
NC_medical_features = pd.read_excel(path, sheet_name='NC-medical')


# Calculate SUVR
for i in range (100):
    AD_image_features.iloc[i, 2:117] /= AD_SUVR_ref.iloc[i,0]
    MCI_image_features.iloc[i, 2:117] /= MCI_SUVR_ref.iloc[i,0]
    NC_image_features.iloc[i, 2:117] /= NC_SUVR_ref.iloc[i,0]

    
# Put all features together
AD = pd.concat([AD_image_features, AD_medical_features.iloc[:,1:]], axis=1)
MCI = pd.concat([MCI_image_features, MCI_medical_features.iloc[:,1:]], axis=1)
NC = pd.concat([NC_image_features, NC_medical_features.iloc[:,1:]], axis=1)

# Male:1 , Female:0
gender_map = {'M': 1, 'F': 0}
AD['Sex'] = AD['Sex'].replace(gender_map)
MCI['Sex'] = MCI['Sex'].replace(gender_map)
NC['Sex'] = NC['Sex'].replace(gender_map)


'''
col 0 -> Subjects
col 1 -> Group
col 2:20 -> subcortical SUVR
col 21:116 -> cortical SUVR
col 117:135 -> subcortical volume
col 136:231 -> cortical volume
col 232 -> MHPSYCH
col 233 -> MH2NEURL
col 234 -> MH4CARD
col 235 -> MMSCORE
col 236 -> CLINICAL DEMENTIA RATING
'''


# Mean & SD
AD_mean = AD.mean(numeric_only=True)
MCI_mean = MCI.mean(numeric_only=True)
NC_mean = NC.mean(numeric_only=True)

AD_SD = AD.std(numeric_only=True)
MCI_SD = MCI.std(numeric_only=True)
NC_SD = NC.std(numeric_only=True)


#%% Feature selection

#AD_new = AD.iloc[:,[0,1,3, 4, 9, 14, 40, 44, 46, 50, 52, 60, 61, 62, 79, 80, 82, 90, 96, 110, 112,124, 125, 134, 139, 150, 151, 203, 232,233,234,235,236]]
#MCI_new = MCI.iloc[:,[0,1,3, 4, 9, 14, 40, 44, 46, 50, 52, 60, 61, 62, 79, 80, 82, 90, 96, 110, 112,124, 125, 134, 139, 150, 151, 203, 232,233,234,235,236]]
#NC_new = NC.iloc[:,[0,1,3, 4, 9, 14, 40, 44, 46, 50, 52, 60, 61, 62, 79, 80, 82, 90, 96, 110, 112,124, 125, 134, 139, 150, 151, 203, 232,233,234,235,236]]

temp1 = pd.DataFrame(AD.mean(numeric_only=True))
temp2 = pd.DataFrame(MCI.mean(numeric_only=True))
temp3 = pd.DataFrame(NC.mean(numeric_only=True))

#drop left and right whole crebral cortex for the sake of plotting
temp1 = temp1.drop('1_left_cerebral_cortex_SUVR,')
temp1 = temp1.drop('11_right_cerebral_cortex_SUVR,')
temp1 = temp1.drop('11_right_cerebral_cortex_volume,')
temp1 = temp1.drop('1_left_cerebral_cortex_volume,')
temp2 = temp2.drop('1_left_cerebral_cortex_SUVR,')
temp2 = temp2.drop('11_right_cerebral_cortex_SUVR,')
temp2 = temp2.drop('11_right_cerebral_cortex_volume,')
temp2 = temp2.drop('1_left_cerebral_cortex_volume,')
temp3 = temp3.drop('1_left_cerebral_cortex_SUVR,')
temp3 = temp3.drop('11_right_cerebral_cortex_SUVR,')
temp3 = temp3.drop('11_right_cerebral_cortex_volume,')
temp3 = temp3.drop('1_left_cerebral_cortex_volume,')


# Define larger font sizes
plt.rcParams.update({
    'font.size': 14,        # General font size
    'axes.titlesize': 16,   # Title font size
    'axes.labelsize': 14,   # Axes labels font size
    'xtick.labelsize': 12,  # X-axis tick labels font size
    'ytick.labelsize': 12,  # Y-axis tick labels font size
    'legend.fontsize': 12   # Legend font size
})


plt.plot(temp1.iloc[0:113], label='AD', color='red', marker='o',linestyle='None')
plt.plot(temp2.iloc[0:113], label='MCI', color='orange', marker='o',linestyle='None')
plt.plot(temp3.iloc[0:113], label='NC', color='green', marker='o',linestyle='None')
plt.xlabel('ROI')
plt.ylabel('SUVR')
plt.title('Average SUVR Mean Comparison')
#plt.legend()  # Add legend
#plt.xticks(rotation='vertical')
plt.show()


plt.plot(temp1.iloc[115:228], label='AD', color='red', marker='o',linestyle='None')
plt.plot(temp2.iloc[115:228], label='MCI', color='orange', marker='o',linestyle='None')
plt.plot(temp3.iloc[115:228], label='NC', color='green', marker='o',linestyle='None')
plt.xlabel('ROI')
plt.ylabel('Volume (mm$^3$)')
plt.title('Average Volume Mean Comparison')
plt.legend()  # Add legend
#plt.xticks(rotation='vertical')
plt.show()



#%% Classification

start_time = time.time()



# Assuming device availability
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


# Hyperparameters
num_epochs = 100
batch_size = 8
shuffle = True

# Concatenating vertically assuming AD, MCI, NC are DataFrames with the same structure
df = pd.concat([AD, MCI, NC], axis=0, ignore_index=True)
df.drop('Subject', axis=1, inplace=True)  # remove subject names

# Scale the features
scaler = MinMaxScaler(feature_range=(0, 1))
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])


class CustomDataset(Dataset):
    def __init__(self, data):
        self.features = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float32)  # Exclude first two columns
        label_encoder = LabelEncoder()
        self.labels = torch.tensor(label_encoder.fit_transform(data.iloc[:, 0]), dtype=torch.long)  # Encode labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Splitting data into train-validation and test sets
#train_val_ratio = 0.9
#test_ratio = 0.1  # 10% of the data for testing
#train_val_data, test_data = train_test_split(df, test_size=test_ratio, random_state=42)

'''
These value names are plural! it means they contain values from all folds.
For example:
    fold_accuracy_lr -> the single accuracy value of that fold
    fold_accuracies_lr -> all accuracy values 
'''
kf = KFold(n_splits=5, shuffle=True, random_state=27)
fold_accuracies_lr = []
fold_accuracies_mlp = []
fold_sensitivities_lr = []
fold_sensitivities_mlp = []
fold_specificities_lr = []
fold_specificities_mlp = []
fold_f1_scores_lr = []
fold_f1_scores_mlp = []


input_size = len(df.columns) - 1  # Subtracting 1 for the label column
num_classes = 3  # Number of classes

test_accuracies = []  # Corrected variable name

# Define logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)
    
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.fc5 = nn.Linear(hidden_size4, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x



hidden_size1 = 64
hidden_size2 = 64
hidden_size3 = 32
hidden_size4 = 32


'''
We have 100 epochs and 5 folds. therefore, every variable needs to have 5 empty lists.
Each list is dedicated to each epoch.
During the epochs, all epoch results are saved in the corrisponding list.
The average value of each list would be the result of that fold.
HOWEVER !!!
To plot accuracy and ROC curves, we need to calculate the point-to-point average 
of all 5 lists. So we have a final list with 100 components to plot.
'''

# Lists to store accuracy and loss values
fold_number = 5
train_accuracy_lr = [[] for _ in range(fold_number)]
train_accuracy_mlp = [[] for _ in range(fold_number)] 
train_loss_lr = [[] for _ in range(fold_number)]
train_loss_mlp = [[] for _ in range(fold_number)] 
val_accuracy_lr = [[] for _ in range(fold_number)]
val_accuracy_mlp = [[] for _ in range(fold_number)]
val_sensitivity_lr = [[] for _ in range(fold_number)]
val_specificity_lr = [[] for _ in range(fold_number)]
val_sensitivity_mlp = [[] for _ in range(fold_number)]
val_specificity_mlp = [[] for _ in range(fold_number)]
fold_all_true_labels = [[] for _ in range(fold_number)]
fold_all_probs_lr = [[] for _ in range(fold_number)]
fold_all_probs_mlp = [[] for _ in range(fold_number)]
    

current_fold = 0; # We have 5 folds

for fold, (train_index, val_index) in enumerate(kf.split(df)):
    train_data, test_data = df.iloc[train_index], df.iloc[val_index]

    train_dataset = CustomDataset(train_data)
    test_dataset = CustomDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize the model
    model_lr = LogisticRegression(input_size, num_classes)
    model_lr.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_lr = optim.Adam(model_lr.parameters(), lr=0.008)  # Adjust learning rate as needed
    scheduler_lr = lr_scheduler.StepLR(optimizer_lr, step_size=10, gamma=0.98) 
    
    # Initialize the model
    model_mlp = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, num_classes)
    model_mlp.to(device)

    # Define  optimizer
    optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=0.005)  # Adjust learning rate as needed
    scheduler_mlp = lr_scheduler.StepLR(optimizer_mlp, step_size=10, gamma=0.98) 
    
   


    # Train the model
    for epoch in range(num_epochs):
        # Training phase
        model_lr.train()
        model_mlp.train()
        epoch_tr_correct_lr = 0
        epoch_tr_total_lr = 0
        epoch_tr_loss_lr = 0.0
        epoch_tr_correct_mlp = 0
        epoch_tr_total_mlp = 0
        epoch_tr_loss_mlp = 0.0

        val_correct_lr = 0
        val_total_lr = 0
        val_correct_mlp = 0
        val_total_mlp = 0

        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Moved to device
            # Forward pass
            tr_outputs_lr = model_lr(inputs)
            tr_outputs_mlp = model_mlp(inputs)
            tr_loss_lr = criterion(tr_outputs_lr, labels)
            tr_loss_mlp = criterion(tr_outputs_mlp, labels)

            # Backward pass and optimization
            optimizer_lr.zero_grad()
            optimizer_mlp.zero_grad()
            tr_loss_lr.backward()
            tr_loss_mlp.backward()
            optimizer_lr.step()
            optimizer_mlp.step()
            scheduler_lr.step()
            scheduler_mlp.step()
            
            # Accuracy
            _, tr_predicted_lr = torch.max(tr_outputs_lr.data, 1)
            epoch_tr_total_lr += labels.size(0)
            epoch_tr_correct_lr += (tr_predicted_lr == labels).sum().item()
            
            _, tr_predicted_mlp = torch.max(tr_outputs_mlp.data, 1)
            epoch_tr_total_mlp += labels.size(0)
            epoch_tr_correct_mlp += (tr_predicted_mlp == labels).sum().item()
            
            
            # Loss
            epoch_tr_loss_lr += tr_loss_lr.item() * inputs.size(0)
            epoch_tr_loss_mlp += tr_loss_mlp.item() * inputs.size(0)

        # Calculate epoch accuracy and loss
        epoch_tr_accuracy_lr = 100 * epoch_tr_correct_lr / epoch_tr_total_lr
        epoch_tr_accuracy_mlp = 100 * epoch_tr_correct_mlp / epoch_tr_total_mlp
        epoch_tr_loss_lr /= len(train_loader.dataset)
        epoch_tr_loss_mlp /= len(train_loader.dataset)
        
        # Store accuracy and loss values
        train_accuracy_lr[current_fold].append(epoch_tr_accuracy_lr)
        train_accuracy_mlp[current_fold].append(epoch_tr_accuracy_mlp)
        train_loss_lr[current_fold].append(epoch_tr_loss_lr)
        train_loss_mlp[current_fold].append(epoch_tr_loss_mlp)

        # Validation phase
        # Evaluate the model on the test set for this fold
        model_lr.eval()
        model_mlp.eval()
        
        
        # Initialize TP_lr, TN_lr, FP_lr, FN_lr for each class
        TP_lr = [0] * num_classes
        TN_lr = [0] * num_classes
        FP_lr = [0] * num_classes
        FN_lr = [0] * num_classes
        
        TP_mlp = [0] * num_classes
        TN_mlp = [0] * num_classes
        FP_mlp = [0] * num_classes
        FN_mlp = [0] * num_classes
 
        
        '''
        You can use sensitivity and specificity values to calculate the ROC curve and AUC, 
        but these values alone do not provide the full set of information needed to generate an ROC curve. 
        The ROC curve requires the true positive rate (sensitivity) and the false positive rate (1 - specificity) 
        at various threshold settings, which is typically derived from the predicted probabilities rather than 
        just a single set of sensitivity and specificity values.
        '''
        all_true_labels = []
        all_probs_lr = []
        all_probs_mlp = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Moved to device
                val_outputs_lr = model_lr(inputs)
                val_outputs_mlp = model_mlp(inputs)
                
                _, val_predicted_lr = torch.max(val_outputs_lr.data, 1)
                val_total_lr += labels.size(0)
                val_correct_lr += (val_predicted_lr == labels).sum().item()
                
                _, val_predicted_mlp = torch.max(val_outputs_mlp.data, 1)
                val_total_mlp += labels.size(0)
                val_correct_mlp += (val_predicted_mlp == labels).sum().item()
                

                for c in range(num_classes):
                    TP_lr[c] += torch.sum((val_predicted_lr == labels) & (val_predicted_lr == c)).item()
                    TN_lr[c] += torch.sum((val_predicted_lr != c) & (labels != c)).item()
                    FP_lr[c] += torch.sum((val_predicted_lr == c) & (labels != c)).item()
                    FN_lr[c] += torch.sum((val_predicted_lr != c) & (labels == c)).item()
                    
                    TP_mlp[c] += torch.sum((val_predicted_mlp == labels) & (val_predicted_mlp == c)).item()
                    TN_mlp[c] += torch.sum((val_predicted_mlp != c) & (labels != c)).item()
                    FP_mlp[c] += torch.sum((val_predicted_mlp == c) & (labels != c)).item()
                    FN_mlp[c] += torch.sum((val_predicted_mlp != c) & (labels == c)).item()
 


                    # Store true labels for ROC
                    all_true_labels.append(labels.cpu().numpy())

                    # Store predicted probabilities for ROC
                    all_probs_lr.append(val_outputs_lr.cpu().numpy())
                    all_probs_mlp.append(val_outputs_mlp.cpu().numpy())

                

        epoch_sensitivity_lr = {}
        epoch_specificity_lr = {}
        epoch_precision_lr = {}
        epoch_f1_score_lr = {}
        
        epoch_sensitivity_mlp = {}
        epoch_specificity_mlp = {}
        epoch_precision_mlp = {}
        epoch_f1_score_mlp = {}
        
        
        for c in range(num_classes):
            epoch_sensitivity_lr[c] = TP_lr[c] / (TP_lr[c] + FN_lr[c]) if (TP_lr[c] + FN_lr[c]) != 0 else 0
            epoch_specificity_lr[c] = TN_lr[c] / (TN_lr[c] + FP_lr[c]) if (TN_lr[c] + FP_lr[c]) != 0 else 0
            epoch_precision_lr[c] = TP_lr[c] / (TP_lr[c] + FP_lr[c]) if (TP_lr[c] + FP_lr[c]) != 0 else 0
            epoch_f1_score_lr[c] = 2 * (epoch_precision_lr[c] * epoch_sensitivity_lr[c]) / (epoch_precision_lr[c] + epoch_sensitivity_lr[c]) if (epoch_precision_lr[c] + epoch_sensitivity_lr[c]) != 0 else 0
            
            epoch_sensitivity_mlp[c] = TP_mlp[c] / (TP_mlp[c] + FN_mlp[c]) if (TP_mlp[c] + FN_mlp[c]) != 0 else 0
            epoch_specificity_mlp[c] = TN_mlp[c] / (TN_mlp[c] + FP_mlp[c]) if (TN_mlp[c] + FP_mlp[c]) != 0 else 0
            epoch_precision_mlp[c] = TP_mlp[c] / (TP_mlp[c] + FP_mlp[c]) if (TP_mlp[c] + FP_mlp[c]) != 0 else 0
            epoch_f1_score_mlp[c] = 2 * (epoch_precision_mlp[c] * epoch_sensitivity_mlp[c]) / (epoch_precision_mlp[c] + epoch_sensitivity_mlp[c]) if (epoch_precision_mlp[c] + epoch_sensitivity_mlp[c]) != 0 else 0

        
        # Calculate overall (macro average) sensitivity, specificity, and F1 score
        epoch_overall_sensitivity_lr = sum(epoch_sensitivity_lr.values()) / num_classes
        epoch_overall_specificity_lr = sum(epoch_specificity_lr.values()) / num_classes
        epoch_overall_f1_score_lr = sum(epoch_f1_score_lr.values()) / num_classes
        
        epoch_overall_sensitivity_mlp = sum(epoch_sensitivity_mlp.values()) / num_classes
        epoch_overall_specificity_mlp = sum(epoch_specificity_mlp.values()) / num_classes
        epoch_overall_f1_score_mlp = sum(epoch_f1_score_mlp.values()) / num_classes


        val_sensitivity_lr[current_fold].append(epoch_sensitivity_lr) # for plotting ROC curve
        val_specificity_lr[current_fold].append(epoch_specificity_lr)
        val_sensitivity_mlp[current_fold].append(epoch_sensitivity_mlp)
        val_specificity_mlp[current_fold].append(epoch_specificity_mlp)

        fold_sensitivity_lr = epoch_overall_sensitivity_lr  # the final epoch sensitivity
        fold_specificity_lr = epoch_overall_specificity_lr  # the final epoch specificity
        fold_f1_score_lr = epoch_overall_f1_score_lr # the final epoch F11 score
        fold_sensitivity_mlp = epoch_overall_sensitivity_mlp
        fold_specificity_mlp = epoch_overall_specificity_mlp 
        fold_f1_score_mlp = epoch_overall_f1_score_mlp
        
        

        # Calculate accuracy
        epoch_val_accuracy_lr = 100 * val_correct_lr / val_total_lr # Gets better and better every epoch
        epoch_val_accuracy_mlp = 100 * val_correct_mlp / val_total_mlp # Gets better and better every epoch
        val_accuracy_lr[current_fold].append(epoch_val_accuracy_lr) #for plotting vs epoch
        val_accuracy_mlp[current_fold].append(epoch_val_accuracy_mlp) #for plotting vs epoch
        
        fold_accuracy_lr = epoch_val_accuracy_lr # The final epoch accuracy
        fold_accuracy_mlp = epoch_val_accuracy_mlp # The final epoch accuracy
        
        
        # Concatenate all batches for ROC
        all_true_labels = np.concatenate(all_true_labels)
        all_probs_lr = np.concatenate(all_probs_lr)
        all_probs_mlp = np.concatenate(all_probs_mlp)
        
        # Save each fold for plotting ROC
        fold_all_true_labels[current_fold] = all_true_labels
        fold_all_probs_lr[current_fold] = all_probs_lr
        fold_all_probs_mlp[current_fold] = all_probs_mlp
    
    
        # END OF EPOCH
    
    fold_accuracies_lr.append(fold_accuracy_lr)
    fold_accuracies_mlp.append(fold_accuracy_mlp)
    
    fold_sensitivities_lr.append(fold_sensitivity_lr)
    fold_specificities_lr.append(fold_specificity_lr)
    fold_f1_scores_lr.append(fold_f1_score_lr)
    fold_sensitivities_mlp.append(fold_sensitivity_mlp)
    fold_specificities_mlp.append(fold_specificity_mlp)
    fold_f1_scores_mlp.append(fold_f1_score_mlp)
    

    
    
        
    print('Accuracy of the LR_model on the test set for fold {}: {:.2f}%'.format(fold+1, fold_accuracy_lr))
    print('Accuracy of the MLP_model on the test set for fold {}: {:.2f}%'.format(fold+1, fold_accuracy_mlp))
    print(' ')
    print('Sensitivity of the LR_model on the test set for fold {}: {:.2f}%'.format(fold+1, fold_sensitivity_lr*100))
    print('Sensitivity of the MLP_model on the test set for fold {}: {:.2f}%'.format(fold+1, fold_sensitivity_mlp*100))
    print(' ')
    print('Specificity of the LR_model on the test set for fold {}: {:.2f}%'.format(fold+1, fold_specificity_lr*100))
    print('Specificity of the MLP_model on the test set for fold {}: {:.2f}%'.format(fold+1, fold_specificity_mlp*100))
    print(' ')
    print('F1 score of the LR_model on the test set for fold {}: {:.2f}%'.format(fold+1, fold_f1_score_lr*100))
    print('F1 score of the MLP_model on the test set for fold {}: {:.2f}%'.format(fold+1, fold_f1_score_mlp*100))


    print('---')
    
    current_fold += 1 #We go to next fold
    
    
    # Save the models
    if current_fold == 5:
        joblib.dump(model_lr, 'model_lr.joblib')
        joblib.dump(model_mlp, 'model_mlp.joblib')
        
        
# Calculate and print the average measures accross all folds
average_accuracy_lr = sum(fold_accuracies_lr) / len(fold_accuracies_lr)
print('Average accuracy of LR model across all folds: {:.2f}%'.format(average_accuracy_lr))
average_accuracy_mlp = sum(fold_accuracies_mlp) / len(fold_accuracies_mlp)
print('Average accuracy of MLP model across all folds: {:.2f}%'.format(average_accuracy_mlp))
print(' ')

average_sensitivity_lr = sum(fold_sensitivities_lr) / len(fold_sensitivities_lr)
print('Average sensitivity of LR model across all folds: {:.2f}%'.format(average_sensitivity_lr*100))
average_sensitivity_mlp = sum(fold_sensitivities_mlp) / len(fold_sensitivities_mlp)
print('Average sensitivity of MLP model across all folds: {:.2f}%'.format(average_sensitivity_mlp*100))
print(' ')

average_specificity_lr = sum(fold_specificities_lr) / len(fold_specificities_lr)
print('Average specificity of LR model across all folds: {:.2f}%'.format(average_specificity_lr*100))
average_specificity_mlp = sum(fold_specificities_mlp) / len(fold_specificities_mlp)
print('Average specificity of MLP model across all folds: {:.2f}%'.format(average_specificity_mlp*100))
print('  ')

average_f1_scores_lr = sum(fold_f1_scores_lr) / len(fold_f1_scores_lr)
print('Average F1 score of LR model across all folds: {:.2f}%'.format(average_f1_scores_lr*100))
average_f1_scores_mlp = sum(fold_specificities_mlp) / len(fold_specificities_mlp)
print('Average F1 score of MLP model across all folds: {:.2f}%'.format(average_f1_scores_mlp*100))
    

# Record the end time
end_time = time.time()
# Calculate the duration
duration = end_time - start_time
print('===')
print("Time taken:", duration, "seconds")




#%% Plotting


# Define larger font sizes
plt.rcParams.update({
    'font.size': 14,        # General font size
    'axes.titlesize': 16,   # Title font size
    'axes.labelsize': 14,   # Axes labels font size
    'xtick.labelsize': 12,  # X-axis tick labels font size
    'ytick.labelsize': 12,  # Y-axis tick labels font size
    'legend.fontsize': 12   # Legend font size
})


# calculate the point-to-point average of variables for all folds
total_accuracy_lr = np.sum(val_accuracy_lr, 0) / 5
total_accuracy_mlp = np.sum(val_accuracy_mlp, 0) / 5
total_train_loss_lr = np.sum(train_loss_lr, 0) / 5
total_train_loss_mlp = np.sum(train_loss_mlp, 0) / 5


fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot accuracy vs Epoch
axs[0].plot(range(1, num_epochs + 1), total_accuracy_lr, label='MLR Model', marker='x',linestyle='-', color = 'green')
axs[0].plot(range(1, num_epochs + 1), total_accuracy_mlp, label='MLP Model', marker='x',linestyle='-', color = 'red')
#axs[0].set_title('Accuracy vs. Epoch')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy (%)')
axs[0].legend()

# Plot loss
axs[1].plot(range(1, num_epochs + 1), total_train_loss_lr, label='MLR Model', marker='x',linestyle='-', color = 'green')
axs[1].plot(range(1, num_epochs + 1), total_train_loss_mlp, label='MLP Model', marker='x',linestyle='-', color = 'red')
axs[1].set_title('Loss vs. Epoch')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].legend()

plt.tight_layout()
plt.show()



# ROC curve

'''
We cannot average the labels!
Moreover, the length of the FPR and TPR arrays may differ for each class and each fold due to differences 
in the dataset or the classifier's predictions.
Therefore, we have to plot them separately.
'''

# Function to plot ROC curve for each class
def plot_roc_curve(y_true, y_probs, model_name, fold):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curve for each class
    plt.figure()
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name} (Fold {fold+1})')
    plt.legend(loc="lower right")
    plt.show()

# One-hot encode the true labels
fold_all_true_labels_oh =  [[] for _ in range(fold_number)]
for fold in range(5):
    fold_all_true_labels_oh[fold] = label_binarize(fold_all_true_labels[fold], classes=[0, 1, 2])

# Plot ROC curve and calculate AUC for each fold and each model
for fold in range(5):
    plot_roc_curve(fold_all_true_labels_oh[fold], fold_all_probs_lr[fold], "MLR Model", fold)
    plot_roc_curve(fold_all_true_labels_oh[fold], fold_all_probs_mlp[fold], "MLP Model", fold)    
    
    
    
    
# barplot for results
# Data preparation
models = ['MLR', 'MLP']
metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'F1 score']

# Values in percentage
data = {
    'MLR': [83.00, 83, 92, 83],
    'MLP': [87.67, 88, 94, 94]
}

# Converting to list of lists for plotting
MLR_values = data['MLR']
MLP_values = data['MLP']

# Setting the positions and width for the bars
bar_width = 0.2
r1 = np.arange(len(MLR_values))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plotting the bars
plt.figure(figsize=(15, 7))

bars1 = plt.bar(r1, MLR_values, color='blue', width=bar_width, edgecolor='grey', label='MLR')
bars2 = plt.bar(r2, MLP_values, color='green', width=bar_width, edgecolor='grey', label='MLP')

# Adding the xticks
plt.xlabel('Metrics')
plt.ylabel('Percentage (%)')
plt.xticks([r + bar_width for r in range(len(MLR_values))], metrics)

# Adding the legend
plt.legend()

# Adding the legend and positioning it above the bars
plt.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=3)

# Adding values on top of the bars
#for bars in [bars1, bars2, bars3]:
for bars in [bars1, bars2]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2 - 0.05, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

# Display the plot
plt.title('Model Performance Comparison')
plt.show()

