# anomaly_transformer.py
import torch
import torch.nn as nn

class AnomalyTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_layers, num_heads):
        super(AnomalyTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, input_dim)  # reconstruct input
        
    def forward(self, src):
        # src shape: (seq_length, batch_size, input_dim)
        transformer_output = self.transformer_encoder(src)
        output = self.fc(transformer_output)
        return output

if __name__ == "__main__":
    # Testing the module with random data
    model = AnomalyTransformer(input_dim=10, model_dim=32, num_layers=2, num_heads=4)
    dummy_input = torch.rand(16, 32, 10)  # (sequence length, batch, features)
    output = model(dummy_input)
    print(output.shape)  # should be (16, 32, 10)
