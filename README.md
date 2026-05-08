# Datascience_Final
## Geographical genomics classification task 

### Overview 
The pipeline involves 5 experiments. Each experiment utilises a random forest classifier in attempts to predict geographical location of samples using kmer features. 
The workflow includes data preprocessing, class imbalance handling, feature selection, optimised tuning and a hierarchical approach. 

### Requirements 
bash
pip install pandas scikit-learn imbalanced-learn seaborn matplotlib

### How to run 
python PredictionModels.py

## Method
### Processing 
- k-mer datasets are transposed 
- Metadata aligned using accession IDs
- Country spelling mistakes are corrected
- Train/test feature alignment applied

- ### Feature
- k-mer frequency vectors
- Feature space reduced using:
  - `SelectKBest (chi-square)`
 
  - ### Experimental designs
  
  - #### Experiment 1: Baseline Models
- Random Forest
- No class balancing
- Separate models for:
  - Country prediction
  - Region prediction


#### Experiment 2: Class Imbalance Handling
- Class weighting used (`class_weight='balanced'`)
- Random oversampling applied using `RandomOverSampler`


#### Experiment 3: Feature Selection Study
- Oversampled dataset
- Chi-square feature selection tested from:
  - kmer = 1,000 to 30,000 features


#### Experiment 4: Hyperparameter Optimisation
- Feature set fixed at k = 30,000
- Tuned parameters tested:
  - Number of trees: 100, 500, 1000
  - Tree depth: 10, 20, 50, None


#### Experiment 5: Hierarchical Model
Two-stage model:

  First stage: Region model predicts geographic region
  
  Second stage: Region prediction is used as an additional feature. Final country model trained using the extra feature space
 

## Visualisation

Confusion matrices were generated for:

- Baseline models
- Best-performing tuned models
- Hierarchical model


