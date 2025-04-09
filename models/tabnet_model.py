# tabnet_model.py
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

def get_tabnet_model(input_dim, output_dim):
    model = TabNetClassifier(
        input_dim=input_dim,
        output_dim=output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='sparsemax'  # "sparsemax" or "entmax"
    )
    return model

if __name__ == '__main__':
    # Example: input_dim=5, output_dim=2 for binary classification
    model = get_tabnet_model(input_dim=5, output_dim=2)
    print(model)
