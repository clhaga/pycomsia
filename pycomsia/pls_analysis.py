import logging

import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# from sklearn.metrics import r2_score
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PLSAnalysis:
    def __init__(self):
        # Original attributes
        self.X_train = None
        self.X_pred = None
        self.y = None
        self.optimal_n_components = None
        self.q2_scores = None
        self.pls_model = None
        self.scalers = None
        self.field_names = None
        self.field_shape = None
        self.field_stdevs = {}
        self.contribution_fractions = None
        self.X_train_final = None  # Training set after split
        self.X_test_final = None  # Test set after split
        self.y_train_final = None  # Training activities after split
        self.y_test_final = None  # Test activities after split
        self.y_train_predicted = None  # Predictions on training set
        self.y_test_predicted = None  # Predictions on test set
        self.y_train_predicted_original = (
            None  # Predictions on training set non-mean centered
        )
        self.y_test_predicted_original = (
            None  # Predictions on test set non-mean centered
        )
        self.y_test_final_original = None  # Test activities non-mean centered
        self.y_train_final_original = None  # Training activities non-mean centered
        self.train_indices = None
        self.test_indices = None
        self.r2_train = None  # R² for training set
        self.r2_test = None  # R² for test set

        # Metrics
        self.spress = None
        self.s_train = None
        self.s_test = None

        # For filtered indices
        self.kept_indices = None
        self.filtered_indices = None

        # Optional prediction set results
        self.y_pred_unseen = None

    def calculate_contribution_fractions(self):
        """
        Calculate contribution fractions for each field based on the final model
        using absolute values of coefficients
        """
        if self.pls_model is None:
            raise ValueError("PLS model has not been fitted yet")

        contribution_fractions = {}
        start_idx = 0

        for field_name in self.field_names:
            # Get number of points kept for this field
            n_kept_points = len(self.kept_indices[field_name])

            # Extract coefficients for this field (only kept points)
            field_coeffs = self.pls_model.coef_[
                0, start_idx : start_idx + n_kept_points
            ]

            # Calculate absolute sum for this field
            field_abs_sum = np.sum(np.abs(field_coeffs))

            # Store the sum
            contribution_fractions[field_name] = field_abs_sum

            # Update start index for next field
            start_idx += n_kept_points

        # Convert to fractions
        total_abs_sum = sum(contribution_fractions.values())
        for field_name in contribution_fractions:
            contribution_fractions[field_name] /= total_abs_sum

        # Store and print results
        self.contribution_fractions = contribution_fractions

        logger.debug("\nField Contribution Fractions:")
        for field_name, fraction in contribution_fractions.items():
            logger.debug(f"{field_name}: {fraction:.4f}")

    def export_predictions_and_residuals(self, outputdir):
        """Export actual values, predicted values, and residuals for both training and test sets to CSV files"""
        if self.y_train_final is None or self.y_train_predicted is None:
            raise ValueError("Model has not been fitted yet")

        # Calculate residuals for training set
        train_residuals = self.y_train_predicted_original - self.y_train_final_original

        # Create DataFrame for training set
        train_df = pd.DataFrame(
            {
                "Set": ["Training"] * len(self.y_train_final_original),
                "Index": self.train_indices,
                "Actual Values": self.y_train_final_original,
                "Predicted Values": self.y_train_predicted_original,
                "Residuals": train_residuals,
            }
        )

        # Calculate residuals for test set
        test_residuals = self.y_test_predicted_original - self.y_test_final_original

        # Create DataFrame for test set
        test_df = pd.DataFrame(
            {
                "Set": ["Test"] * len(self.y_test_final_original),
                "Index": self.test_indices,
                "Actual Values": self.y_test_final_original,
                "Predicted Values": self.y_test_predicted_original,
                "Residuals": test_residuals,
            }
        )

        # Combine training and test sets
        combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

        # Export combined DataFrame to CSV file
        combined_df.to_csv(
            f"{outputdir}/PLS_Analysis/Predictions_and_Residuals.csv", index=False
        )

        # Export separate files for training and test sets
        train_df.to_csv(
            f"{outputdir}/PLS_Analysis/Training_Predictions_and_Residuals.csv",
            index=False,
        )
        test_df.to_csv(
            f"{outputdir}/PLS_Analysis/Test_Predictions_and_Residuals.csv", index=False
        )

        logger.info(f"Residual files exported to {outputdir}/PLS_Analysis/")

    def export_metrics_to_csv(self, outputdir):
        """Export all metrics to a CSV file"""
        metrics = {
            "r2_train": self.r2_train,
            "r2_test": self.r2_test,
            "q2": np.max(self.q2_scores),
            "SPRESS": self.spress,
            "S_train": self.s_train,
            "S_test": self.s_test,
            "Number of Components": self.optimal_n_components,
        }

        # Add contribution fractions
        for field_name, fraction in self.contribution_fractions.items():
            metrics[f"Contribution Fraction ({field_name})"] = fraction

        # Convert to DataFrame and export
        df = pd.DataFrame([metrics])
        df.to_csv(f"{outputdir}/PLS_Analysis/PLS_Metrics.csv", index=False)

    def convert_fields_to_X(self, train_fields, pred_fields=None, filter=0.0):
        """
        Convert fields to X matrix with filtering and scaling.
        """
        combined_fields = {}
        n_train_molecules = len(train_fields["steric_field"])

        for field_name in train_fields:
            train_data = np.array(train_fields[field_name])
            # Ensure 2D array (molecules x features)
            if train_data.ndim == 1:
                train_data = train_data.reshape(1, -1)

            if pred_fields is not None:
                pred_data = np.array(pred_fields[field_name])
                if pred_data.ndim == 1:
                    pred_data = pred_data.reshape(1, -1)
                combined_fields[field_name] = np.vstack((train_data, pred_data))
            else:
                combined_fields[field_name] = train_data

        self.kept_indices = {}
        self.filtered_indices = {}
        filtered_fields = {}

        for field_name in combined_fields:
            ranges = np.ptp(combined_fields[field_name], axis=0)
            kept_idx = np.where(ranges >= filter)[0]
            filtered_idx = np.where(ranges < filter)[0]

            self.kept_indices[field_name] = kept_idx
            self.filtered_indices[field_name] = filtered_idx

            filtered_fields[field_name] = combined_fields[field_name][:, kept_idx]

            logger.debug(f"\n{field_name}:")
            logger.debug(f"Original points: {len(ranges)}")
            logger.debug(f"Points kept: {len(kept_idx)}")
            logger.debug(f"Points filtered: {len(filtered_idx)}")

        X_train = []
        X_pred = None if pred_fields is None else []
        self.scalers = {}
        self.field_stdevs = {}  # Initialize field stdevs

        for field_name in filtered_fields:
            field_data = filtered_fields[field_name]

            # Scale COMBINED data
            self.scalers[field_name] = StandardScaler()
            scaled_data = self.scalers[field_name].fit_transform(
                field_data.reshape(-1, 1)
            )
            scaled_data = scaled_data.reshape(field_data.shape)
            self.field_stdevs[field_name] = np.std(
                scaled_data[:n_train_molecules]
            )  # Calculate stdev on training set only

            # Split AFTER scaling
            scaled_train = scaled_data[:n_train_molecules]
            scaled_pred = (
                scaled_data[n_train_molecules:] if pred_fields is not None else None
            )

            if len(X_train) == 0:
                X_train = scaled_train
            else:
                X_train = np.hstack((X_train, scaled_train))

            if pred_fields is not None:
                if len(X_pred) == 0:
                    X_pred = scaled_pred
                else:
                    X_pred = np.hstack((X_pred, scaled_pred))

        # Mean center by column (do this after combining all fields)
        X_train_means = np.mean(X_train, axis=0)
        X_train = X_train - X_train_means

        if X_pred is not None:
            X_pred_means = np.mean(X_pred, axis=0)
            X_pred = X_pred - X_pred_means

        self.X_train = X_train
        self.X_pred = X_pred
        self.field_names = list(train_fields.keys())

        logger.debug(f"X_train shape: {X_train.shape}")
        if X_pred is not None:
            logger.debug(f"X_pred shape: {X_pred.shape}")

        # Determine field shape based on data structure
        first_field = train_fields[self.field_names[0]]
        if isinstance(first_field[0], (int, float, np.number)):
            # Single array case - field_shape is total length divided by number of fields
            self.field_shape = (len(first_field) // len(self.field_names),)
        else:
            # List of arrays case
            self.field_shape = (len(first_field[0]),)

        return X_train, X_pred

    def get_coefficient_fields(self):
        """
        Reconstruct full field arrays from final PLS model coefficients
        maintaining the scaled coefficients and applying standard deviation scaling
        """
        if self.pls_model is None:
            raise ValueError("PLS model has not been fitted yet")

        reconstructed_fields = {}
        contour_values = {}  # Store contour values (coefficients * standard deviations)
        start_idx = 0

        for field_name in self.field_names:
            # Get number of points kept for this field
            n_kept_points = len(self.kept_indices[field_name])

            # Extract coefficients for this field (only for kept points)
            field_coeffs_kept = self.pls_model.coef_[
                0, start_idx : start_idx + n_kept_points
            ]

            # Create full-size array of zeros
            field_coeffs_full = np.zeros(self.field_shape[0])

            # Place the coefficients in their original positions
            field_coeffs_full[self.kept_indices[field_name]] = field_coeffs_kept

            # Get the full standard deviation array
            field_std_full = np.zeros(self.field_shape[0])
            field_std_full[self.kept_indices[field_name]] = self.field_stdevs[
                field_name
            ]

            # Multiply coefficients by standard deviations to get contour values
            contour_values_full = field_coeffs_full * field_std_full

            # Store the coefficients and contour values
            reconstructed_fields[field_name] = field_coeffs_full
            contour_values[field_name] = contour_values_full

            nonzero = np.count_nonzero(contour_values_full)
            logger.debug(
                f"  {field_name}: {n_kept_points} kept pts, "
                f"coeff range=[{field_coeffs_kept.min():.4f}, {field_coeffs_kept.max():.4f}], "
                f"{nonzero} non-zero contour pts"
            )

            # Update start index for next field
            start_idx += n_kept_points

        # Return contour values
        return contour_values

    def perform_loo_analysis(self, train_activities, max_components=12):
        """Perform Leave-One-Out cross-validation analysis to determine optimal components"""
        loo = LeaveOneOut()
        y = np.array(train_activities)
        y_mean = np.mean(y)
        y = y - y_mean
        q2_scores = []
        press_values_all_components = []
        n_mols = self.X_train.shape[0]
        max_comp_actual = min(n_mols, max_components + 1)
        logger.debug(
            f"LOO CV: {n_mols} molecules, testing {max_comp_actual - 1} components"
        )

        total_fits = (max_comp_actual - 1) * n_mols
        with tqdm(
            total=total_fits,
            desc="LOO CV",
            unit="fit",
            leave=True,
        ) as pbar:
            for n_components in range(1, max_comp_actual):
                press_values = []
                ypreds = []
                press_values = []
                ssy_values = []

                for train_index, test_index in loo.split(self.X_train):
                    X_train_loo, X_test_loo = (
                        self.X_train[train_index],
                        self.X_train[test_index],
                    )
                    y_train_loo, y_test_loo = y[train_index], y[test_index]

                    pls = PLSRegression(
                        n_components=n_components, scale=False, tol=1e-4, max_iter=100
                    )
                    pls.fit(X_train_loo, y_train_loo)
                    y_pred = pls.predict(X_test_loo)[0]
                    press_values.append((y_test_loo[0] - y_pred) ** 2)
                    ssy_values.append((y_test_loo[0]) ** 2)
                    ypreds.append(y_pred)
                    pbar.update(1)

                press_values_all_components.append(np.array(press_values))
                mean_press = np.mean(press_values)
                mean_ssy = np.mean(ssy_values)
                q2 = 1 - (mean_press / mean_ssy)

                logger.debug(f"q2 = {q2:.4f} for component = {n_components}")
                pbar.set_postfix({"comp": n_components, "q2": f"{q2:.3f}"})
                q2_scores.append(q2)

        self.optimal_n_components = np.argmax(q2_scores) + 1
        self.q2_scores = q2_scores
        self.press_values = np.array(press_values_all_components)
        self.spress = np.sqrt(np.mean(self.press_values[self.optimal_n_components - 1]))
        logger.info(
            f"Optimal number of components based on maximum Q2: {self.optimal_n_components}"
        )

    def fit_final_model(
        self, train_activities, test_size=0.2, predict_smiles_list=None
    ):
        """
        Split data into train/test, fit final model with optimal components on training set,
        and evaluate on test set
        """
        # Split into train and test sets
        from sklearn.model_selection import train_test_split

        y = np.array(train_activities)
        y_mean = np.mean(y)
        y = y - y_mean  # Mean center y

        # Generate indices for the entire dataset
        indices = np.arange(len(self.X_train))

        # Split the indices into train and test sets
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=44
        )

        # Store the train and test indices
        self.train_indices = train_indices
        self.test_indices = test_indices

        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            self.X_train, y, test_size=test_size, random_state=44
        )

        # Store the split data
        self.X_train_final = X_train_split
        self.X_test_final = X_test_split
        self.y_train_final = y_train_split
        self.y_test_final = y_test_split

        # Fit final model on training set
        self.pls_model = PLSRegression(
            n_components=self.optimal_n_components, scale=False, tol=1e-4, max_iter=100
        )
        self.pls_model.fit(X_train_split, y_train_split)

        # Get predictions for both sets
        self.y_train_predicted = self.pls_model.predict(X_train_split).ravel()
        self.y_test_predicted = self.pls_model.predict(X_test_split).ravel()

        # Reverse mean-centering for predictions and test values
        self.y_train_final_original = self.y_train_final + y_mean
        self.y_train_predicted_original = self.y_train_predicted + y_mean
        self.y_test_predicted_original = self.y_test_predicted + y_mean
        self.y_test_final_original = self.y_test_final + y_mean

        # Calculate R² for both sets (using original values)

        self.r2_train = r2_score(
            self.y_train_final_original, self.y_train_predicted_original
        )
        logger.info(f"R² Training: {self.r2_train:.4f}")
        if self.y_test_final is not None:
            self.r2_test = r2_score(
                self.y_test_final_original, self.y_test_predicted_original
            )
        logger.info(f"R² Test: {self.r2_test:.4f}")

        # Calculate SPESS and S for training set
        residuals_train = self.y_train_final_original - self.y_train_predicted_original
        self.s_train = np.std(residuals_train)

        # Calculate SPESS and S for test set if it exists
        if self.y_test_final is not None:
            residuals_test = self.y_test_final_original - self.y_test_predicted_original
            self.s_test = np.std(residuals_test)

        # Print SPESS and S values
        logger.info(f"SPRESS: {self.spress:.4f}")
        logger.info(f"S Training: {self.s_train:.4f}")
        if self.y_test_final is not None:
            logger.info(f"S Test: {self.s_test:.4f}")
        self.calculate_contribution_fractions()

        # Make predictions for unseen molecules if prediction set exists
        if self.X_pred is not None:
            self.y_pred_unseen = self.pls_model.predict(self.X_pred).ravel()
            self.y_pred_unseen_original = (
                self.y_pred_unseen + y_mean
            )  # Reverse mean-centering for unseen predictions
            df = pd.DataFrame(
                {
                    "SMILES": predict_smiles_list,
                    "Predicted Activity": self.y_pred_unseen_original,
                }
            )
            df.to_csv("predictions.csv", index=False)

    def plot_results(self, outputdir):
        """Generate plots for analysis results"""
        plt.rcParams.update(
            {"font.size": 16, "axes.titlesize": 20, "axes.labelsize": 18}
        )

        fig = plt.figure(figsize=(15, 10))

        # Plot 1: Q2 scores
        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(self.q2_scores) + 1), self.q2_scores, "bo-")
        plt.xlabel("Number of Components")
        plt.ylabel("Q² Score")
        plt.title("Q² Scores vs Number of Components")

        # Calculate R² for both sets
        r2_train = self.r2_train
        r2_test = self.r2_test

        # Plot 2: Actual vs predicted values
        plt.subplot(2, 2, 2)
        # Plot training set in blue
        sns.scatterplot(
            x=self.y_train_final_original,
            y=self.y_train_predicted_original,
            label="Training Set",
            s=100,
            color="blue",
        )
        # Plot test set in red
        sns.scatterplot(
            x=self.y_test_final_original,
            y=self.y_test_predicted_original,
            label="Test Set",
            s=100,
            color="red",
        )

        if self.y_pred_unseen is not None:
            plt.axhline(
                y=np.min(self.y_pred_unseen), color="gray", linestyle="--", alpha=0.3
            )
            plt.axhline(
                y=np.max(self.y_pred_unseen), color="gray", linestyle="--", alpha=0.3
            )

        # Plot perfect prediction line
        all_y = np.concatenate(
            [self.y_train_final_original, self.y_test_final_original]
        )
        plt.plot(
            [min(all_y), max(all_y)],
            [min(all_y), max(all_y)],
            "k--",
            label="Perfect Prediction",
        )

        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values")

        # Add R² text for both sets
        plt.text(
            0.95,
            0.05,
            f"R²(train) = {self.r2_train:.4f}",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            horizontalalignment="right",
        )  # Align text to the right

        plt.text(
            0.95,
            0.12,
            f"R²(test) = {self.r2_test:.4f}",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            horizontalalignment="right",
        )

        plt.legend()

        # Calculate residuals for both sets
        train_residuals = self.y_train_predicted_original - self.y_train_final_original
        test_residuals = self.y_test_predicted_original - self.y_test_final_original

        # Plot 3: Residuals vs Predicted
        plt.subplot(2, 2, 3)
        # Plot training set residuals in blue
        sns.scatterplot(
            x=self.y_train_predicted_original,
            y=train_residuals,
            label="Training Set",
            s=100,
            color="blue",
        )
        # Plot test set residuals in red
        sns.scatterplot(
            x=self.y_test_predicted_original,
            y=test_residuals,
            label="Test Set",
            s=100,
            color="red",
        )
        plt.axhline(y=0, color="k", linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Predicted Values")
        plt.legend()

        # Plot 4: Residuals Distribution
        plt.subplot(2, 2, 4)
        sns.histplot(
            train_residuals, kde=True, color="blue", alpha=0.5, label="Training Set"
        )
        sns.histplot(test_residuals, kde=True, color="red", alpha=0.5, label="Test Set")
        plt.xlabel("Residuals")
        plt.ylabel("Count")
        plt.title("Distribution of Residuals")
        plt.legend()

        plt.tight_layout()

        filename = f"{outputdir}/PLS_Analysis/PLSplots.png"
        plt.savefig(filename)

        # Print statistics for both sets
        logger.info("\nModel Statistics:")
        logger.info("Training Set:")
        train_rmse = np.sqrt(np.mean(train_residuals**2))
        train_mae = np.mean(np.abs(train_residuals))
        logger.info(f"R² Score: {r2_train:.4f}")
        logger.info(f"Root Mean Square Error: {train_rmse:.4f}")
        logger.info(f"Mean Absolute Error: {train_mae:.4f}")
        logger.info(f"Std of Residuals: {np.std(train_residuals):.4f}")

        logger.info("\nTest Set:")
        test_rmse = np.sqrt(np.mean(test_residuals**2))
        test_mae = np.mean(np.abs(test_residuals))
        logger.info(f"R² Score: {r2_test:.4f}")
        logger.info(f"Root Mean Square Error: {test_rmse:.4f}")
        logger.info(f"Mean Absolute Error: {test_mae:.4f}")
        logger.info(f"Std of Residuals: {np.std(test_residuals):.4f}")
