# models/anomaly_transformer.py
import torch
import torch.nn as nn

class AnomalyTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_layers, num_heads):
        super(AnomalyTransformer, self).__init__()
        # Project input from input_dim (e.g., 10) to model_dim (32)
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, input_dim)  # Project back to original dimension if needed

    def forward(self, src):
        # src: shape (seq_length, batch, input_dim)
        # Project input to model dimension
        src = self.input_proj(src)  # Now shape: (seq_length, batch, model_dim)
        transformer_output = self.transformer_encoder(src)
        output = self.fc(transformer_output)
        return output

if __name__ == "__main__":
    # Test with dummy data
    model = AnomalyTransformer(input_dim=10, model_dim=32, num_layers=2, num_heads=4)
    dummy_input = torch.rand(16, 32, 10)  # (sequence length, batch, input_dim)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Should be (16, 32, 10)
