import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xgboost as xgb
from anomaly_transformer import AnomalyTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

class FraudDetectionPipeline:
    def __init__(self, ts_input_dim, tab_input_dim, transformer_epochs=50):
        # Initialize the anomaly transformer model.
        self.anomaly_model = AnomalyTransformer(input_dim=ts_input_dim, model_dim=32, num_layers=2, num_heads=4)
        self.xgb_model = None  # XGBoost model (to be trained later)
        self.transformer_epochs = transformer_epochs
        self.mse_loss = nn.MSELoss()
    
    def train_anomaly_model(self, ts_data, optimizer, epochs):
        """Train the anomaly transformer using MSE loss on the time-series data."""
        self.anomaly_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            anomaly_out = self.anomaly_model(ts_data)
            loss = self.mse_loss(anomaly_out, ts_data)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Anomaly Transformer - Epoch {epoch}: Loss = {loss.item():.6f}")
    
    def train(self, ts_data, tab_data, labels):
        """
        Trains both the anomaly transformer and the XGBoost model.
          • ts_data shape: (seq_length, batch_size, ts_features)
          • tab_data shape: (batch_size, tab_input_dim)
          • labels: binary fraud labels for each sample (batch_size,)
        """
        # Train the anomaly transformer.
        optimizer = optim.Adam(self.anomaly_model.parameters(), lr=1e-3)
        print("Training Anomaly Transformer...")
        self.train_anomaly_model(ts_data, optimizer, self.transformer_epochs)
        
        # Compute anomaly reconstruction error for each sample.
        self.anomaly_model.eval()
        with torch.no_grad():
            anomaly_out = self.anomaly_model(ts_data)
            # Compute error by averaging over sequence and feature dimensions.
            anomaly_error = torch.mean(torch.mean((ts_data - anomaly_out) ** 2, dim=2), dim=0).cpu().numpy()
        
        # Combine tabular data with the anomaly error feature.
        # X shape will become (batch_size, tab_input_dim + 1)
        X = np.hstack((tab_data, anomaly_error.reshape(-1, 1)))
        y = labels
        
        # Split data into training and validation sets.
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train an XGBoost classifier.
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'eta': 0.1,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
        print("Training XGBoost model...")
        evals = [(dtrain, 'train'), (dval, 'eval')]
        self.xgb_model = xgb.train(params, dtrain, num_boost_round=100, evals=evals, early_stopping_rounds=10, verbose_eval=10)
    
    def predict(self, ts_data, tab_data):
        """
        Generates predictions by computing the anomaly error from the anomaly transformer and
        combining it (as a feature) with the tabular data to run through the XGBoost model.
        """
        self.anomaly_model.eval()
        with torch.no_grad():
            anomaly_out = self.anomaly_model(ts_data)
            anomaly_error = torch.mean(torch.mean((ts_data - anomaly_out)**2, dim=2), dim=0).cpu().numpy()
        
        X = np.hstack((tab_data, anomaly_error.reshape(-1, 1)))
        dtest = xgb.DMatrix(X)
        xgb_preds = self.xgb_model.predict(dtest)  # Returns fraud probabilities
        return xgb_preds

if __name__ == '__main__':
    # Simulated Data for Testing
    # Time-series data: shape (sequence_length, batch_size, ts_features)
    seq_length, batch_size, ts_features = 16, 320, 10  # Using 320 samples for better stability
    ts_data = torch.rand(seq_length, batch_size, ts_features)
    
    # Tabular data: shape (batch_size, tab_input_dim), e.g., 5 features per sample.
    tab_input_dim = 5
    tab_data = np.random.rand(batch_size, tab_input_dim)
    
    # Simulated binary labels for each sample.
    labels = np.random.randint(0, 2, batch_size)
  
    pipeline = FraudDetectionPipeline(ts_input_dim=ts_features, tab_input_dim=tab_input_dim, transformer_epochs=50)
    
    print("Starting training...")
    pipeline.train(ts_data, tab_data, labels)
    
    print("Generating predictions...")
    preds = pipeline.predict(ts_data, tab_data)
    
    auc = roc_auc_score(labels, preds)
    print("ROC AUC:", auc)
