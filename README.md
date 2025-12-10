# COMP0213 – Grasp Planning Project (Group 9)

This repository implements a complete robotic grasp planning framework using **PyBullet**, **machine learning classification**, and a fully modular simulation pipeline. The system simulates multiple gripper types, samples grasp poses around various object geometries, evaluates grasp stability in physics, and trains high-level classifiers to predict grasp success.

The implementation is cleanly modularised and organised to support dataset generation, training, and testing in a reproducible workflow.

---

## Installation

Install all dependencies from the requirements.txt file

This installs:

- PyBullet + pybullet_data  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- scikit-learn  
---

## Running the Pipeline

Navigate into the `src/` directory:


Inside `main.py`, you can enable or disable stages of the pipeline depending on what you want to run.

### Generate dataset (optional)
Uncomment:

```python
pipeline.generate_dataset()
```

This will:

-sample grasp poses around objects
-simulate grasp execution
-store success/failure labels
-save datasets to CSV or append to existing CSVs

### Train classification models
```python
pipeline.train_classifiers()
```
This step:

-loads datasets from the CSV files
-balances success vs. failure samples
-trains Logistic Regression, SVM, Random Forest, Gradient Boosting, and MLP models
-performs hyperparameter tuning if enabled
-selects and saves the best-performing model per gripper/object combination

Savied models appear under 
`src/trained_models/`

### Test trained classifiers
```python
pipeline.test_classifiers()
```

This will:

generate new test grasps
-predict success using trained classifiers
-simulate each grasp in PyBullet
-report accuracy and confusion matrices
-generate and save performance plots

NOTE: The testing and training functions do not work independently, as the plots require the training statistics
The model will not be retrained if use_saved_models is set to True and force_retrain is set to False in the config dictionary.


### Configuration
Edit `get_config()` in `main.py` to adjust parameters.
```python
"gui": False,                        # Enable PyBullet GUI
"sampling_radius": 0.4,              # Spherical sampling radius
"noise_std": 0.03,                   # Positional noise in sampling
"samples_per_combination": 100,      # Number of grasps per dataset
"n_test_samples": 50,                # Test grasps per classifier
"tune_hyperparams": True,            # Use GridSearchCV for tuning
"use_saved_models": True,            # Load saved .pkl models if present
"force_retrain": False               # Retrain even if saved models exist
```
