import os
from typing import Dict, List, Tuple

import numpy as np
import pybullet as p

from classifier.GraspClassifier import GraspClassifier
from data.DataManager import DataManager
from grippers.ThreeFingerGripper import ThreeFingerGripper
from grippers.TwoFingerGripper import TwoFingerGripper
from objects.BoxObject import BoxObject
from objects.CylinderObject import CylinderObject
from sampling.SphericalSampler import SphericalSampler
from simulation.GraspSimulator import GraspSimulator


class GraspPipeline:
    """Full pipeline: dataset generation, training, and testing."""

    def __init__(self, config: Dict):
        self.config = config
        self.simulator = GraspSimulator(gui=config["gui"])

        self.data_managers = {
            "TwoFinger_Box": DataManager(
                filename=config["dataset_file_two_finger_box"]
            ),
            "TwoFinger_Cylinder": DataManager(
                filename=config["dataset_file_two_finger_cylinder"]
            ),
            "ThreeFinger_Box": DataManager(
                filename=config["dataset_file_three_finger_box"]
            ),
            "ThreeFinger_Cylinder": DataManager(
                filename=config["dataset_file_three_finger_cylinder"]
            ),
        }

        tune = config.get("tune_hyperparams", False)
        self.classifiers = {
            "TwoFinger_Box": GraspClassifier("TwoFinger", "Box", tune_hyperparams=tune),
            "TwoFinger_Cylinder": GraspClassifier(
                "TwoFinger", "Cylinder", tune_hyperparams=tune
            ),
            "ThreeFinger_Box": GraspClassifier(
                "ThreeFinger", "Box", tune_hyperparams=tune
            ),
            "ThreeFinger_Cylinder": GraspClassifier(
                "ThreeFinger", "Cylinder", tune_hyperparams=tune
            ),
        }

        self.train_metrics: Dict[str, Dict] = {}
        self.test_metrics: Dict[str, Dict] = {}

    # Dataset generation
    def generate_dataset(self):
        """ generate datasets for all gripper+object combinations """
        print("\n" + "=" * 70)
        print("DATASET GENERATION")
        print("=" * 70)

        self.simulator.connect()

        combinations = [
            (TwoFingerGripper(), BoxObject([0, 0, 0.05]), "TwoFinger", "Box"),
            (
                TwoFingerGripper(),
                CylinderObject([0, 0, 0.1], self.config["cylinder_urdf"]),
                "TwoFinger",
                "Cylinder",
            ),
            (
                ThreeFingerGripper(self.config["three_finger_urdf"]),
                BoxObject([0, 0, 0.05]),
                "ThreeFinger",
                "Box",
            ),
            (
                ThreeFingerGripper(self.config["three_finger_urdf"]),
                CylinderObject([0, 0, 0.1], self.config["cylinder_urdf"]),
                "ThreeFinger",
                "Cylinder",
            ),
        ]

        sampler = SphericalSampler(
            radius=self.config["sampling_radius"],
            noise_std=self.config["noise_std"],
        )

        for gripper, obj, gripper_name, obj_name in combinations:
            key = f"{gripper_name}_{obj_name}"
            print(f"\n{key}")
            print("-" * 70)

            self._collect_data(
                gripper,
                obj,
                gripper_name,
                obj_name,
                sampler,
                self.config["samples_per_combination"],
                self.data_managers[key],
            )

            self.data_managers[key].save_to_csv()
            print(f"Saved to {self.data_managers[key].filename}")

        self.simulator.disconnect()
        print("\n" + "=" * 70)

    def _collect_data(
        self,
        gripper,
        obj,
        gripper_name: str,
        obj_name: str,
        sampler: SphericalSampler,
        n_samples: int,
        data_manager: DataManager,
    ):
        """ collect all the data needed for sampling """
        gripper.load()
        gripper.attach_fixed()
        gripper.open()
        self.simulator.step(20)

        if isinstance(gripper, ThreeFingerGripper):
            gripper.set_default_position()
            self.simulator.step(20)

        obj_pos = np.array(obj.position)

        for i in range(n_samples):
            if (i + 1) % 50 == 0:
                print(f"Sample {i + 1}/{n_samples}")

            position, orientation = sampler.sample_pose(obj_pos, obj)

            if isinstance(gripper, TwoFingerGripper) and isinstance(obj, CylinderObject):
                orientation[1] -= np.pi / 2
            elif isinstance(gripper, ThreeFingerGripper) and isinstance(obj, BoxObject):
                orientation[1] += np.pi / 2

            success = self.simulator.execute_grasp(
                gripper, obj, position, orientation
            )
            data_manager.add_sample(position, orientation, success, gripper_name, obj_name)

        try:
            obj.remove()
        except Exception:
            pass

        try:
            if gripper.id is not None:
                p.removeBody(gripper.id)
                gripper.id = None
        except Exception:
            pass

        self.simulator.step(10)

    # Dataset analysis

    def analyze_dataset_distribution(self):
        """ analyse dataset and check the average success """
        print("\n" + "=" * 70)
        print("DATASET ANALYSIS")
        print("=" * 70)

        for key, manager in self.data_managers.items():
            df = manager.load_dataset()
            if len(df) > 0:
                success_rate = df["success"].mean()
                print(f"{key:25s}: {len(df)} samples, {success_rate:.1%}")

    # Training

    def train_classifiers(self):
        """ train classifier models to find desired hyperparameters """
        print("\n" + "=" * 70)
        print("MODEL TRAINING PHASE")
        print("=" * 70)

        use_saved = self.config.get("use_saved_models", False)
        force_retrain = self.config.get("force_retrain", False)

        for key in self.classifiers.keys():
            df = self.data_managers[key].load_dataset()

            if len(df) == 0:
                print(f"✗ No data available for {key}")
                continue

            df_balanced = self.data_managers[key].get_balanced_dataset(df)
            metrics = self.classifiers[key].train(
                df_balanced,
                use_saved=use_saved,
                force_retrain=force_retrain,
            )
            self.train_metrics[key] = metrics

    # Testing

    def test_classifiers(self):
        """ test trained models with new data to see how they perform """
        print("\n" + "=" * 70)
        print("MODEL TESTING PHASE")
        print("=" * 70)

        # Load models if they haven't been trained in this session
        for key, classifier in self.classifiers.items():
            if classifier.model is None:
                print(f"\nLoading model for {key}...")
                if not classifier.load_model():
                    raise ValueError(
                        f"No trained model found for {key}. "
                        f"Please train models first or check model path."
                    )

        test_configs = [
            (TwoFingerGripper(), BoxObject([0, 0, 0.05]), "TwoFinger", "Box"),
            (
                TwoFingerGripper(),
                CylinderObject([0, 0, 0.1], self.config["cylinder_urdf"]),
                "TwoFinger",
                "Cylinder",
            ),
            (
                ThreeFingerGripper(self.config["three_finger_urdf"]),
                BoxObject([0, 0, 0.05]),
                "ThreeFinger",
                "Box",
            ),
            (
                ThreeFingerGripper(self.config["three_finger_urdf"]),
                CylinderObject([0, 0, 0.1], self.config["cylinder_urdf"]),
                "ThreeFinger",
                "Cylinder",
            ),
        ]

        for gripper, obj, gripper_name, obj_name in test_configs:
            key = f"{gripper_name}_{obj_name}"
            print(f"\n{key}")
            print("-" * 70)
            self._run_test(gripper, obj, gripper_name, obj_name)

        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATION PLOTS")
        print("=" * 70)

        for key in self.classifiers.keys():
            if key in self.test_metrics:
                print(f"\nGenerating plot for {key}...")
                self.classifiers[key].visualize_results(
                    self.train_metrics[key],
                    self.test_metrics[key],
                    save_path=f"results_{key}.png",
                )

    def _run_test(self, gripper, obj, gripper_name: str, obj_name: str):
        """ run a test to see and evalute error metrics """
        self.simulator.connect()

        gripper.load()
        gripper.attach_fixed()

        if isinstance(gripper, ThreeFingerGripper):
            gripper.set_default_position()
            self.simulator.step(20)

        gripper.open()
        self.simulator.step(20)

        sampler = SphericalSampler(
            radius=self.config["sampling_radius"],
            noise_std=self.config["noise_std"],
        )
        obj_pos = np.array(obj.position)

        key = f"{gripper_name}_{obj_name}"
        classifier = self.classifiers[key]
        n_tests = self.config["n_test_samples"]

        predictions: List[bool] = []
        actuals: List[bool] = []

        for _ in range(n_tests):
            position, orientation = sampler.sample_pose(obj_pos, obj)

            if isinstance(gripper, TwoFingerGripper) and isinstance(obj, CylinderObject):
                orientation[1] -= np.pi / 2
            elif isinstance(gripper, ThreeFingerGripper) and isinstance(obj, BoxObject):
                orientation[1] += np.pi / 2

            pred = classifier.predict(position, orientation)
            actual = self.simulator.execute_grasp(gripper, obj, position, orientation)

            predictions.append(pred)
            actuals.append(actual)

        self.simulator.disconnect()

        predictions_arr = np.array(predictions)
        actuals_arr = np.array(actuals)

        from sklearn.metrics import confusion_matrix

        accuracy = (predictions_arr == actuals_arr).mean()
        cm = confusion_matrix(actuals_arr, predictions_arr)

        self.test_metrics[key] = {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "predictions": predictions_arr,
            "actuals": actuals_arr,
        }

        print(f"Test accuracy: {accuracy:.1%}")
        print(f"Actual success rate: {actuals_arr.mean():.1%}")
        print(f"Predicted success rate: {predictions_arr.mean():.1%}")

    def run_full_pipeline(self):
        """ run the entire pipeline by using all the function previously defined """
        self.generate_dataset()
        self.analyze_dataset_distribution()
        self.train_classifiers()
        self.test_classifiers()
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)