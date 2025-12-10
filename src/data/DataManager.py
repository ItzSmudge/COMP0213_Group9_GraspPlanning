from typing import Dict, List

import os
import pandas as pd


class DataManager:
    """Handle creation, storage, and loading of grasp datasets."""

    def __init__(self, filename: str = "grasp_dataset.csv"):
        """ initialise instance of class. Set the filename and set up a space for the data"""
        self.filename = filename
        self.data: List[Dict] = []

    def add_sample(
        self,
        position: List[float],
        orientation: List[float],
        success: bool,
        gripper_type: str,
        object_type: str
    ) -> None:
        """ add the necessary features to the data dictionary """
        self.data.append({
            "x": position[0],
            "y": position[1],
            "z": position[2],
            "roll": orientation[0],
            "pitch": orientation[1],
            "yaw": orientation[2],
            "success": 1 if success else 0,
            "gripper": gripper_type,
            "object": object_type
        })

    def save_to_csv(self) -> None:
        "Add the sample created of the features to the csv "
        if not self.data:
            print("No data to save!")
            return

        df = pd.DataFrame(self.data)

        if os.path.exists(self.filename):
            df.to_csv(self.filename, mode="a", header=False, index=False)
        else:
            df.to_csv(self.filename, index=False)

        print(f"Saved {len(self.data)} samples to {self.filename}")
        self.data.clear()

    def load_dataset(self) -> pd.DataFrame:
        """ load the dataset by accessing the csv"""
        if not os.path.exists(self.filename) or os.path.getsize(self.filename) == 0:
            print(f"[INFO] No dataset found at {self.filename}. Creating new...")
            return pd.DataFrame(columns=[
                "x", "y", "z", "roll", "pitch", "yaw", "success", "gripper", "object"
            ])
        return pd.read_csv(self.filename)

    def get_balanced_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        "create a new dataset by using the exisiting dataset. Aim to balance the number of successes and failures"
        successes = df[df["success"] == 1]
        failures = df[df["success"] == 0]

        min_count = min(len(successes), len(failures))

        if min_count == 0:
            print("[WARNING] No balanced data available!")
            return df

        balanced_df = pd.concat([
            successes.sample(n=min_count, random_state=42),
            failures.sample(n=min_count, random_state=42)
        ]).sample(frac=1, random_state=42).reset_index(drop=True)

        print(
            f"Balanced dataset: {len(balanced_df)} samples "
            f"({min_count} success, {min_count} failure)"
        )

        return balanced_df
