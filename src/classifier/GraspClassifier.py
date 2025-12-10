from typing import Dict

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


class GraspClassifier:
    def __init__(self, gripper_name=None, object_name=None, tune_hyperparams=False):
        self.gripper_name = gripper_name
        self.object_name = object_name
        self.model = None
        self.best_model_name = None
        self.feature_cols = ["x", "y", "z", "roll", "pitch", "yaw"]
        self.target_col = "success"
        self.tune_hyperparams = tune_hyperparams
        self.model_path = f"trained_models/{gripper_name}_{object_name}.pkl"

        self.models = {
            "logistic": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    C=1.0, max_iter=1000, class_weight="balanced"
                ))
            ]),
            "svm": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(
                    C=1.0, kernel="rbf", probability=True,
                    class_weight="balanced"
                ))
            ]),
            "random_forest": Pipeline([
                ("model", RandomForestClassifier(
                    n_estimators=200, max_depth=12,
                    class_weight="balanced", random_state=42
                ))
            ]),
            "gradient_boosting": Pipeline([
                ("model", GradientBoostingClassifier(
                    n_estimators=150, learning_rate=0.1,
                    max_depth=5, random_state=42
                ))
            ]),
            "neural_network": Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(
                    hidden_layer_sizes=(64, 32), max_iter=500,
                    alpha=0.0001, random_state=42
                ))
            ])
        }

        self.model_parameters = {
            "logistic": {
                "model__C": [0.1, 1.0, 10.0]
            },
            "svm": {
                "model__C": [0.1, 1.0, 10.0],
                "model__gamma": ["scale", "auto"]
            },
            "random_forest": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [8, 12, 16]
            },
            "gradient_boosting": {
                "model__n_estimators": [100, 150, 200],
                "model__learning_rate": [0.05, 0.1, 0.15]
            },
            "neural_network": {
                "model__hidden_layer_sizes": [(32, 16), (64, 32), (128, 64)],
                "model__alpha": [0.0001, 0.001]
            }
        }

    def save_model(self):
        """Save the trained model to disk."""
        os.makedirs("trained_models", exist_ok=True)
        model_data = {
            "model": self.model,
            "best_model_name": self.best_model_name,
            "gripper_name": self.gripper_name,
            "object_name": self.object_name
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"✓ Model saved to: {self.model_path}")

    def load_model(self):
        """Load a previously trained model from disk."""
        if not os.path.exists(self.model_path):
            print(f"✗ No saved model found at: {self.model_path}")
            return False

        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.best_model_name = model_data["best_model_name"]
        print(f"✓ Loaded saved model from: {self.model_path}")
        print(f"  Model type: {self.best_model_name}")
        return True

    def train(self, df, use_saved=False, force_retrain=False):
        """
        Train or load a classifier model.
        """
        X = df[self.feature_cols].values
        y = df[self.target_col].values

        print("\n" + "=" * 70)
        print(f"Training: {self.gripper_name} + {self.object_name}")
        print("=" * 70)
        print(f"Dataset size: {len(df)} samples")
        print(f"Success rate: {y.mean():.1%}")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if use_saved and not force_retrain:
            if self.load_model():
                train_score = self.model.score(X_train, y_train)
                val_score = self.model.score(X_val, y_val)
                print("\nPerformance of loaded model:")
                print(f"  Train accuracy: {train_score:.3f}")
                print(f"  Validation accuracy: {val_score:.3f}")

                return {
                    "X_train": X_train,
                    "X_val": X_val,
                    "y_train": y_train,
                    "y_val": y_val,
                    "train_score": train_score,
                    "val_score": val_score,
                    "all_results": {},
                    "loaded_from_disk": True
                }
            else:
                print("No saved model found. Training new model...\n")

        if force_retrain:
            print("Force retrain enabled. Training new model...\n")

        best_val_score = 0
        results: Dict[str, Dict] = {}

        print("Training all model types:")
        print("-" * 70)

        for name, pipeline in self.models.items():
            if self.tune_hyperparams and name in self.model_parameters:
                print(f"\n{name.upper()} (with hyperparameter tuning)")
                grid_search = GridSearchCV(
                    pipeline,
                    self.model_parameters[name],
                    cv=3,
                    scoring="accuracy",
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train, y_train)
                best_pipeline = grid_search.best_estimator_
                train_score = best_pipeline.score(X_train, y_train)
                val_score = best_pipeline.score(X_val, y_val)
                print(f"  Best params: {grid_search.best_params_}")
                print(f"  Train: {train_score:.3f}, Val: {val_score:.3f}")
            else:
                print(f"{name:20s}", end=" ")
                pipeline.fit(X_train, y_train)
                best_pipeline = pipeline
                train_score = best_pipeline.score(X_train, y_train)
                val_score = best_pipeline.score(X_val, y_val)
                print(f"Train: {train_score:.3f}, Val: {val_score:.3f}")

            results[name] = {
                "train": train_score,
                "val": val_score,
                "model": best_pipeline
            }

            if val_score > best_val_score:
                best_val_score = val_score
                self.model = best_pipeline
                self.best_model_name = name

        print("\n" + "-" * 70)
        print(f"Best model selected: {self.best_model_name.upper()}")
        print(f"  Validation accuracy: {best_val_score:.3f}")
        print(f"  Train accuracy: {results[self.best_model_name]['train']:.3f}")

        self.save_model()

        return {
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val,
            "train_score": results[self.best_model_name]["train"],
            "val_score": results[self.best_model_name]["val"],
            "all_results": results,
            "loaded_from_disk": False
        }

    def predict(self, position, orientation):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a saved model.")
        features = np.array([[*position, *orientation]])
        return bool(self.model.predict(features)[0])

    def predict_proba(self, position, orientation):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a saved model.")
        features = np.array([[*position, *orientation]])
        return self.model.predict_proba(features)[0][1]

    def visualize_results(self, train_metrics, test_metrics, save_path=None):
        has_importance = hasattr(
            self.model.named_steps.get("model"), "feature_importances_"
        )

        if has_importance:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        cm = test_metrics["confusion_matrix"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail", "Success"],
            yticklabels=["Fail", "Success"], ax=axes[0]
        )
        axes[0].set_title(
            f"{self.gripper_name} + {self.object_name}\nConfusion Matrix (Test Set)"
        )
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")

        scores = [
            train_metrics["train_score"],
            train_metrics["val_score"],
            test_metrics["accuracy"]
        ]
        bars = axes[1].bar(["Train", "Validation", "Test"], scores, alpha=0.8)
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim([0, 1.0])
        axes[1].set_title(f"Model Performance\n({self.best_model_name})")
        axes[1].axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.2%}",
                ha="center",
                va="bottom",
                fontweight="bold"
            )

        if has_importance:
            importances = self.model.named_steps["model"].feature_importances_
            indices = np.argsort(importances)[::-1]
            sorted_features = [self.feature_cols[i] for i in indices]
            sorted_importances = importances[indices]

            axes[2].barh(sorted_features, sorted_importances)
            axes[2].set_xlabel("Importance")
            axes[2].set_title("Feature Importance")
            axes[2].invert_yaxis()

            for i, (feat, imp) in enumerate(zip(sorted_features, sorted_importances)):
                axes[2].text(imp + 0.01, i, f"{imp:.3f}", va="center", fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved: {save_path}")

        plt.show()
