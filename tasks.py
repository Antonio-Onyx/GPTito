import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        return F.log_softmax(logits, dim=-1)
    
class RegressionHead(nn.Module):
    def __init__(self, d_model, output_dim=1, activation=None):
        super().__init__()
        self.fc = nn.Linear(d_model, output_dim)
        self.activation = activation

    def forward(self, x):
        output = self.fc(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(output)
        elif self.activation == "relu":
            return F.relu(output)
        return output