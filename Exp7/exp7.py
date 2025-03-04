import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

# Load dataset
heartDisease = pd.read_csv('7-dataset.csv')

# Replace missing values with NaN
heartDisease.replace('?', np.nan, inplace=True)

# Rename "gender" column to "sex" to match Bayesian model
heartDisease.rename(columns={'gender': 'sex'}, inplace=True)

# Convert categorical columns to numeric if necessary
heartDisease['ca'] = pd.to_numeric(heartDisease['ca'], errors='coerce')
heartDisease['thal'] = pd.to_numeric(heartDisease['thal'], errors='coerce')

# Display dataset samples
print('Sample instances from the dataset:')
print(heartDisease.head())

# Display attributes and their data types
print('\nAttributes and datatypes:')
print(heartDisease.dtypes)

# Define Bayesian Network Model
model = BayesianNetwork([
    ('age', 'heartdisease'),
    ('sex', 'heartdisease'),
    ('exang', 'heartdisease'),
    ('cp', 'heartdisease'),
    ('heartdisease', 'restecg'),
    ('heartdisease', 'chol')
])

# Learn CPD using Maximum Likelihood Estimators
print('\nLearning CPD using Maximum Likelihood Estimators...')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

# Perform inference
print('\nInferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)

# Query 1: Probability of HeartDisease given evidence = restecg
print('\n1. Probability of HeartDisease given evidence (restecg=1):')
q1 = HeartDisease_infer.query(variables=['heartdisease'], evidence={'restecg': 1})
print(q1)

# Query 2: Probability of HeartDisease given evidence = cp
print('\n2. Probability of HeartDisease given evidence (cp=2):')
q2 = HeartDisease_infer.query(variables=['heartdisease'], evidence={'cp': 2})
print(q2)
