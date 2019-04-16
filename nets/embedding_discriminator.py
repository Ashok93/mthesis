import torch
import torch.nn as nn


class FeatureDiscriminator(nn.Module):

    def __init__(self):
        super(FeatureDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, embedding):
        model_op = self.model(embedding)
        return model_op
