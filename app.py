import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from typing import Tuple, Optional, Dict, Any
from spark_utils import spark_processor

class AutoMLPipeline:
    def __init__(self):
        self.data = None
        self.model = None
        self.target_column = None
        self.feature_columns = None
        self.model_type = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None
        
    def load_data(self, file_path: str):
        df = spark_processor.load_large_dataset(file_path)
        if df is not None:
            self.data = df
            return self.data, f"Data loaded successfully! Shape: {self.data.shape}"
        else:
            return None, "Error loading data."
    
    def get_data_info(self):
        if spark_processor.df is not None:
            info = spark_processor.get_spark_data_profile()
            return {
                'shape': (info['total_rows'], info['total_cols']),
                'columns': list(spark_processor.df.columns),
                'dtypes': info['dtypes'],
                'missing_values': info['missing_counts'],
                'numeric_columns': [k for k, v in info['dtypes'].items() if 'int' in v or 'double' in v or 'float' in v or 'long' in v],
                'categorical_columns': [k for k, v in info['dtypes'].items() if 'StringType' in v]
            }
        elif self.data is not None:
            return {
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'dtypes': self.data.dtypes.to_dict(),
                'missing_values': self.data.isnull().sum().to_dict(),
                'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(self.data.select_dtypes(include=['object']).columns)
            }
        else:
            return {}
    
    def create_general_overview_plots(self):
        if self.data is None:
            return None, None, None, None, None
        # Missing values heatmap
        fig_missing = px.imshow(
            self.data.isnull().values,
            title="Missing Values Heatmap",
            color_continuous_scale="Reds",
            aspect="auto"
        )
        fig_missing.update_layout(
            xaxis_title="Columns",
            yaxis_title="Rows",
            height=400
        )
        # Data types distribution
        dtype_counts = self.data.dtypes.value_counts()
        fig_dtypes = px.pie(
            values=dtype_counts.values,
            names=dtype_counts.index.astype(str),
            title="Data Types Distribution"
        )
        # Correlation heatmap for numeric columns
        numeric_data = self.data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr()
            fig_corr = px.imshow(
                corr_matrix,
                title="Correlation Matrix",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            fig_corr.update_layout(height=500)
        else:
            fig_corr = go.Figure().add_annotation(
                text="Not enough numeric columns for correlation analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        # Numeric Feature Distributions
        if not numeric_data.empty:
            melted = numeric_data.melt(var_name="Feature", value_name="Value")
            fig_num_dist = px.histogram(melted, x="Value", color="Feature", facet_col="Feature", facet_col_wrap=3, title="Numeric Feature Distributions", nbins=30, opacity=0.7, histnorm=None)
        else:
            fig_num_dist = None
        # Boxplots for Outlier Detection
        if not numeric_data.empty:
            melted = numeric_data.melt(var_name="Feature", value_name="Value")
            fig_box = px.box(melted, x="Feature", y="Value", title="Boxplots for Outlier Detection")
        else:
            fig_box = None
        return fig_missing, fig_dtypes, fig_corr, fig_num_dist, fig_box

    def create_target_dependent_plots(self):
        if self.data is None or self.target_column is None or self.target_column not in self.data.columns:
            return None, None, None
        # Target Variable Distribution
        target_data = self.data[self.target_column].dropna()
        if target_data.dtype == 'object' or len(target_data.unique()) < 20:
            fig_target = px.histogram(target_data, title="Target Variable Distribution", labels={"value": "Target", "count": "Frequency"})
        else:
            fig_target = px.histogram(target_data, nbins=30, title="Target Variable Distribution", labels={"value": "Target", "count": "Frequency"})
        # Feature Correlation with Target
        numeric_data = self.data.select_dtypes(include=[np.number])
        if not numeric_data.empty and self.data[self.target_column].dtype in [np.float64, np.float32, np.int64, np.int32]:
            corrs = []
            for col in numeric_data.columns:
                corr = self.data[col].corr(self.data[self.target_column])
                corrs.append((col, corr))
            if corrs:
                corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)
                fig_corr_target = px.bar(x=[x[0] for x in corrs], y=[x[1] for x in corrs], title="Feature Correlation with Target", labels={"x": "Feature", "y": "Correlation"})
            else:
                fig_corr_target = None
        else:
            fig_corr_target = None
        # Class Imbalance Visualization
        if target_data.dtype == 'object' or len(target_data.unique()) < 20:
            fig_class_imbalance = px.pie(names=target_data, title="Class Imbalance Visualization")
        else:
            fig_class_imbalance = None
        return fig_target, fig_corr_target, fig_class_imbalance
    
    def prepare_data(self, target_column: str) -> str:
        if self.data is None:
            return "No data loaded"
        
        if target_column not in self.data.columns:
            return f"Target column '{target_column}' not found in data"
        
        self.target_column = target_column
        self.feature_columns = [col for col in self.data.columns if col != target_column]
        
        clean_data = self.data.dropna(subset=[target_column])
        
        X = clean_data[self.feature_columns]
        y = clean_data[target_column]
        
        if y.dtype == 'object' or len(y.unique()) < 20:
            self.model_type = 'classification'
        else:
            self.model_type = 'regression'
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return f"Data prepared for {self.model_type}. Training set: {self.X_train.shape}, Test set: {self.X_test.shape}"
    
    def train_model(self) -> str:
        if self.X_train is None:
            return "Data not prepared. Please select target column first."
        
        try:
            if self.model_type == 'classification':
                self.model = CatBoostClassifier(
                    iterations=100,
                    learning_rate=0.1,
                    depth=6,
                    verbose=False,
                    random_seed=42
                )
            else:
                self.model = CatBoostRegressor(
                    iterations=100,
                    learning_rate=0.1,
                    depth=6,
                    verbose=False,
                    random_seed=42
                )
            
            categorical_features = []
            for i, col in enumerate(self.feature_columns):
                if self.X_train[col].dtype == 'object':
                    categorical_features.append(i)
            
            self.model.fit(
                self.X_train,
                self.y_train,
                cat_features=categorical_features,
                eval_set=(self.X_test, self.y_test),
                verbose=False
            )
            
            self.predictions = self.model.predict(self.X_test)
            
            return "Model trained successfully!"
        
        except Exception as e:
            return f"Error training model: {str(e)}"
    
    def get_model_performance(self) -> Tuple[str, Any]:
        if self.model is None or self.predictions is None:
            return "Model not trained yet", None
        
        if self.model_type == 'regression':
            mse = mean_squared_error(self.y_test, self.predictions)
            r2 = r2_score(self.y_test, self.predictions)
            
            metrics_text = f"""
            **Regression Metrics:**
            - Mean Squared Error: {mse:.4f}
            - R² Score: {r2:.4f}
            - RMSE: {np.sqrt(mse):.4f}
            """
            
            fig = px.scatter(
                x=self.y_test,
                y=self.predictions,
                title="Predictions vs Actual Values",
                labels={'x': 'Actual', 'y': 'Predicted'}
            )
            fig.add_shape(
                type="line",
                x0=self.y_test.min(),
                y0=self.y_test.min(),
                x1=self.y_test.max(),
                y1=self.y_test.max(),
                line=dict(dash="dash", color="red")
            )
            
        else:
            accuracy = accuracy_score(self.y_test, self.predictions)
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
            
            precision = precision_score(self.y_test, self.predictions, average=None)
            recall = recall_score(self.y_test, self.predictions, average=None)
            f1 = f1_score(self.y_test, self.predictions, average=None)
            
            classes = np.unique(self.y_test)
            
            metrics_text = f"""
            Classification Metrics:
            - Overall Accuracy: {accuracy:.4f}
            
            Per-Class Metrics:
            """
            
            for i, cls in enumerate(classes):
                metrics_text += f"""
                Class {cls}:
                - Precision: {precision[i]:.4f}
                - Recall: {recall[i]:.4f}
                - F1-Score: {f1[i]:.4f}
                """
            
            metrics_text += f"""
Detailed Classification Report:
```
{classification_report(self.y_test, self.predictions)}
```
"""
            
            cm = confusion_matrix(self.y_test, self.predictions)
            fig = px.imshow(
                cm,
                title="Confusion Matrix",
                color_continuous_scale="Blues",
                aspect="auto",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=classes,
                y=classes
            )
            fig.update_layout(
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                height=500
            )
        
        return metrics_text, fig
    
    def get_feature_importance(self) -> Any:
        """Get feature importance plot"""
        if self.model is None:
            return None
        
        try:
            importance = self.model.get_feature_importance()
            feature_names = self.feature_columns
            
            fig = px.bar(
                x=importance,
                y=feature_names,
                orientation='h',
                title="Feature Importance",
                labels={'x': 'Importance', 'y': 'Features'}
            )
            fig.update_layout(height=max(400, len(feature_names) * 25))
            
            return fig
        except:
            return None

pipeline = AutoMLPipeline()

def process_file(file):
    if file is None:
        return "Please upload a file", "", gr.update(choices=[]), gr.update(interactive=False), None, None, None, None, None
    
    pipeline.data, message = pipeline.load_data(file.name)
    
    if pipeline.data is not None:
        info = pipeline.get_data_info()
        info_text = f"""
        Dataset Information:
        - Shape: {info['shape'][0]} rows, {info['shape'][1]} columns
        - Numeric columns: {len(info['numeric_columns'])}
        - Categorical columns: {len(info['categorical_columns'])}
        - Missing values: {sum(info['missing_values'].values())} total
        """
        # Create general overview plots
        plots = pipeline.create_general_overview_plots()
        return message, info_text, gr.update(choices=info['columns']), gr.update(interactive=True), *plots
    else:
        return message, "", gr.update(choices=[]), gr.update(interactive=False), None, None, None, None, None

def on_target_change(target_column):
    if target_column:
        return gr.update(interactive=True)
    return gr.update(interactive=False)

def update_target_dependent_plots(target_column):
    pipeline.target_column = target_column
    plots = pipeline.create_target_dependent_plots()
    return plots

def create_overview_plots():
    return pipeline.create_data_overview_plots()

def train_pipeline(target_column):
    if not target_column or target_column not in pipeline.data.columns:
        return "Please select a valid target column", "", None, None
    
    prep_message = pipeline.prepare_data(target_column)
    train_message = pipeline.train_model()
    
    metrics_text, performance_plot = pipeline.get_model_performance()
    
    importance_plot = pipeline.get_feature_importance()
    combined_message = f"{prep_message}\n{train_message}"
    
    return combined_message, metrics_text, performance_plot, importance_plot

with gr.Blocks(title="Auto ML Pipeline", theme=gr.themes.Soft()) as app:
    gr.Markdown("#Auto ML Pipeline with CatBoost")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Data Upload")
            file_input = gr.File(
                label="Upload CSV or Excel file",
                file_types=[".csv", ".xlsx", ".xls"]
            )
            upload_status = gr.Textbox(label="Upload Status", interactive=False)
            
            gr.Markdown("## Dataset Information")
            data_info = gr.Textbox(label="Data Overview", interactive=False, lines=8)
            
            gr.Markdown("## Target Selection")
            target_dropdown = gr.Dropdown(
                label="Select Target Column",
                choices=[],
            )
            
            train_button = gr.Button("Train Model", variant="primary", size="lg", interactive=False)
            training_status = gr.Textbox(label="Training Status", interactive=False)
        
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Data Overview"):
                    missing_plot = gr.Plot(label="Missing Values")
                    dtypes_plot = gr.Plot(label="Data Types")
                    corr_plot = gr.Plot(label="Correlation Matrix")
                    num_dist_plot = gr.Plot(label="Numeric Feature Distributions")
                    boxplot = gr.Plot(label="Boxplots for Outlier Detection")
                    target_dist_plot = gr.Plot(label="Target Variable Distribution")
                    corr_target_plot = gr.Plot(label="Feature Correlation with Target")
                    class_imbalance_plot = gr.Plot(label="Class Imbalance Visualization")
                
                with gr.TabItem("Model Performance"):
                    performance_metrics = gr.Markdown("Train a model to see performance metrics")
                    performance_plot = gr.Plot(label="Model Performance")
                
                with gr.TabItem("Feature Importance"):
                    importance_plot = gr.Plot(label="Feature Importance")
    
    file_input.change(
        fn=process_file,
        inputs=[file_input],
        outputs=[upload_status, data_info, target_dropdown, train_button, missing_plot, dtypes_plot, corr_plot, num_dist_plot, boxplot]
    )
    
    target_dropdown.change(
        fn=on_target_change,
        inputs=[target_dropdown],
        outputs=[train_button]
    )
    target_dropdown.change(
        fn=update_target_dependent_plots,
        inputs=[target_dropdown],
        outputs=[target_dist_plot, corr_target_plot, class_imbalance_plot]
    )
    
    train_button.click(
        fn=train_pipeline,
        inputs=[target_dropdown],
        outputs=[training_status, performance_metrics, performance_plot, importance_plot]
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        debug=True
    ) 