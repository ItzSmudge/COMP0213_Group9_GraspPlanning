from pipeline.GraspPipeline import GraspPipeline


def get_config():
    """
    Global configuration for the grasp planning pipeline.
    """
    return {
        "gui": False,
        "dataset_file_two_finger_box": "grasp_dataset_two_finger_box.csv",
        "dataset_file_two_finger_cylinder": "grasp_dataset_two_finger_cylinder.csv",
        "dataset_file_three_finger_box": "grasp_dataset_three_finger_box.csv",
        "dataset_file_three_finger_cylinder": "grasp_dataset_three_finger_cylinder.csv",
        "cylinder_urdf": "urdf/cylinder.urdf",
        "three_finger_urdf": "urdf/sdh.urdf",
        "sampling_radius": 0.4,
        "noise_std": 0.03,
        "samples_per_combination": 500,
        "n_test_samples": 50,       
        "tune_hyperparams": True,   # Use hyperparameter tuning
        "use_saved_models": True,   # Try to load saved models
        "force_retrain": False,     # Set True to retrain and overwrite
    }


def main():
    config = get_config()
    pipeline = GraspPipeline(config)

    # Uncomment any of the below lines depending on purpose
    #pipeline.generate_dataset()   # Uncomment this line to add more data to datasets

    #pipeline.analyze_dataset_distribution()  # Analyze percentage of success/failure in each dataset

    """NOTE: these two functions do not work independently, as the plots require the training statistics
    The model will not be retrained if use_saved_models is set to True and force_retrain is set to False
    in the config dictionary"""
    pipeline.train_classifiers()             #Train classifcation models 
    pipeline.test_classifiers()              #Test classification models on new data


if __name__ == "__main__":
    main()
