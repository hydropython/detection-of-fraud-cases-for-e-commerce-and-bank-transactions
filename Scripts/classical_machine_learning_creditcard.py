import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
import os
import joblib

class CreditCardModelingPipeline:
    def __init__(self, df, target_column, dataset_name="fraud-data"):
        self.df = df
        self.target_column = target_column
        self.dataset_name = dataset_name
        self.X = pd.get_dummies(df.drop(columns=[target_column]), drop_first=True)
        self.y = df[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    
    def train_models(self):
        mlflow.set_experiment(f"{self.dataset_name}_model_training")
        active_run = mlflow.active_run()
        if active_run:
            mlflow.end_run()

        with mlflow.start_run():
            try:
                # Logistic Regression
                self.logistic_model = LogisticRegression(max_iter=2000)
                self.logistic_model.fit(self.X_train, self.y_train)
                self.log_metrics(self.y_test, self.logistic_model.predict(self.X_test), "Logistic Regression")
                mlflow.sklearn.log_model(self.logistic_model, "logistic_regression_model")
                joblib.dump(self.logistic_model, "../models/logistic_regression_model.pkl")  # Save as .pkl

                # Decision Tree
                self.decision_tree_model = DecisionTreeClassifier(random_state=42)
                self.decision_tree_model.fit(self.X_train, self.y_train)
                self.log_metrics(self.y_test, self.decision_tree_model.predict(self.X_test), "Decision Tree")
                mlflow.sklearn.log_model(self.decision_tree_model, "decision_tree_model")
                joblib.dump(self.decision_tree_model, "../models/decision_tree_model.pkl")  # Save as .pkl

                # Random Forest
                self.random_forest_model = RandomForestClassifier(random_state=42)
                self.random_forest_model.fit(self.X_train, self.y_train)
                self.log_metrics(self.y_test, self.random_forest_model.predict(self.X_test), "Random Forest")
                mlflow.sklearn.log_model(self.random_forest_model, "random_forest_model")
                joblib.dump(self.random_forest_model, "../models/random_forest_model.pkl")  # Save as .pkl

                # Gradient Boosting
                self.gradient_boosting_model = GradientBoostingClassifier(random_state=42)
                self.gradient_boosting_model.fit(self.X_train, self.y_train)
                self.log_metrics(self.y_test, self.gradient_boosting_model.predict(self.X_test), "Gradient Boosting")
                mlflow.sklearn.log_model(self.gradient_boosting_model, "gradient_boosting_model")
                joblib.dump(self.gradient_boosting_model, "../models/gradient_boosting_model.pkl")  # Save as .pkl

            except Exception as e:
                print("An error occurred during model training:", e)
                if mlflow.active_run():
                    mlflow.end_run()

    def log_metrics(self, y_true, y_pred, model_name):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"--- {model_name} ---")
        print(classification_report(y_true, y_pred))
        with mlflow.start_run(nested=True):
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            img_path = self.plot_confusion_matrix(y_true, y_pred, model_name)
            mlflow.log_artifact(img_path)
            os.remove(img_path)

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, cmap='Blues')
        plt.title(f"Confusion Matrix for {model_name}")
        plt.colorbar()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.xticks(np.arange(2), ['Not Fraud', 'Fraud'])
        plt.yticks(np.arange(2), ['Not Fraud', 'Fraud'])
        
        img_path = f"confusion_matrix_{model_name}.png"
        plt.savefig(img_path)
        plt.close()
        return img_path

    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning using RandomizedSearchCV for Random Forest and Gradient Boosting"""
        # Random Forest Hyperparameter Tuning
        rf_param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        
        rf_random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_distributions=rf_param_distributions,
            n_iter=50,
            scoring='accuracy',
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        rf_random_search.fit(self.X_train, self.y_train)
        print(f"Best Random Forest Parameters: {rf_random_search.best_params_}")
        self.rf_best_model = rf_random_search.best_estimator_
        rf_predictions = self.rf_best_model.predict(self.X_test)
        
        # Evaluation and logging to MLflow
        self.log_metrics(self.y_test, rf_predictions, "Random Forest (Tuned)")
        mlflow.sklearn.log_model(self.rf_best_model, artifact_path="random_forest_best_model")
        
        # Gradient Boosting Hyperparameter Tuning
        gb_param_distributions = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.05],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.6, 0.8, 1.0]
        }
        
        gb_random_search = RandomizedSearchCV(
            estimator=GradientBoostingClassifier(random_state=42),
            param_distributions=gb_param_distributions,
            n_iter=50,
            scoring='accuracy',
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        gb_random_search.fit(self.X_train, self.y_train)
        print(f"Best Gradient Boosting Parameters: {gb_random_search.best_params_}")
        self.gb_best_model = gb_random_search.best_estimator_
        gb_predictions = self.gb_best_model.predict(self.X_test)
        
        # Evaluation and logging to MLflow
        self.log_metrics(self.y_test, gb_predictions, "Gradient Boosting (Tuned)")
        mlflow.sklearn.log_model(self.gb_best_model, artifact_path="gradient_boosting_best_model")

    def shap_analysis(self):
        """
        Generates a SHAP analysis for the trained gradient boosting model on the test dataset.
        Produces and saves a SHAP summary plot as a PNG file and a SHAP force plot as an HTML file.
        """
        # Check if the gradient boosting model has been trained
        if not hasattr(self, 'gradient_boosting_model'):
            raise AttributeError("The model is not trained. Please train the model by running `train_models()` first.")
        
        # Ensure the output directory exists
        output_dir = '../Images/'
        os.makedirs(output_dir, exist_ok=True)

        # Initialize the SHAP explainer for the gradient boosting model
        explainer = shap.TreeExplainer(self.gradient_boosting_model)

        # Calculate SHAP values for the test set
        shap_values = explainer.shap_values(self.X_test)

        # Determine SHAP values and expected value based on binary or multiclass model
        if isinstance(shap_values, list) and len(shap_values) > 1:  # Binary classification
            shap_values_class = shap_values[1]  # SHAP values for the positive class
            expected_value = explainer.expected_value[1]
        else:  # Multiclass or regression
            shap_values_class = shap_values
            expected_value = explainer.expected_value

        # Generate and save the SHAP summary plot
        plt.figure()
        shap.summary_plot(shap_values_class, self.X_test, plot_type="bar", show=False)  # show=False to suppress plot display
        summary_plot_path = os.path.join(output_dir, "shap_summary_plot.png")
        plt.savefig(summary_plot_path, bbox_inches='tight')  # Save the summary plot as PNG
        plt.close()  # Close the figure

        # Generate and save a SHAP force plot for the first instance
        shap.initjs()  # Initialize JavaScript for interactive SHAP plots
        force_plot = shap.force_plot(expected_value, shap_values_class[0], self.X_test.iloc[0], matplotlib=False)
        force_plot_path = os.path.join(output_dir, "shap_force_plot.html")
        shap.save_html(force_plot_path, force_plot)  # Save the force plot as HTML

        print(f"SHAP summary plot saved as {summary_plot_path}")
        print(f"SHAP force plot saved as {force_plot_path}")
    def lime_analysis(self):
        """Perform LIME analysis to explain model predictions."""
        lime_explainer = LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.X.columns,
            class_names=['Not Fraud', 'Fraud'],
            mode='classification'
        )

        # Select an instance to explain
        i = 0  # Change this index to explain different instances
        exp = lime_explainer.explain_instance(
            data_row=self.X_test.iloc[i].values,
            predict_fn=self.gradient_boosting_model.predict_proba
        )

        # Save LIME explanation as an image
        plt.figure()
        exp.as_pyplot_figure()
        plt.savefig('../Images/lime_explanation.png')  # Save as PNG
        plt.close()

        print("LIME explanation saved as lime_explanation.png")