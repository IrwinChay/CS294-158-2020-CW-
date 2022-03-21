import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torchlib.common import FloatTensor
from torchlib.dataset.utils import create_data_loader
from torchlib.generative_model.made import MADE


class WarmUpModel(nn.Module):
    def __init__(self, n=100):
        super(WarmUpModel, self).__init__()
        self.n = n
        self.theta = nn.Parameter(torch.randn(1, self.n))

    def forward(self, x):
        return self.theta.repeat((x.shape[0], 1))

    @property
    def pmf(self):
        return F.softmax(self.theta[0].cpu().detach(), dim=-1).numpy()

    def sample(self, shape):
        p = self.pmf
        return np.random.choice(np.arange(self.n), size=shape, p=p)


class MLP(nn.Module):
    def __init__(self, n, nn_size=32, n_layers=3):
        super(MLP, self).__init__()
        self.n = n
        self.embedding = nn.Embedding(n, nn_size)
        models = []
        models.append(self.embedding)
        models.append(nn.Dropout(0.5))
        for i in range(n_layers - 1):
            models.append(nn.Linear(nn_size, nn_size))
            models.append(nn.ReLU())
        models.append(nn.Linear(nn_size, n))
        self.model = nn.Sequential(*models)

    def forward(self, x1):
        """

        Args:
            x1: The condition variable x1. of shape (batch_size). Encoded as one hot vector.

        Returns: a logits over x2

        """
        return self.model.forward(x1)


class TwoDimensionModel(nn.Module):
    def __init__(self, n=200):
        super(TwoDimensionModel, self).__init__()
        self.x2_cond_x1 = MLP(n=n)
        self.x1_model = WarmUpModel(n=n)

    def forward(self, x):
        x1 = x[:, 0]
        return self.x1_model.forward(x1), self.x2_cond_x1.forward(x1)

    def sample(self, num_samples):
        self.eval()
        with torch.no_grad():
            x1 = self.x1_model.sample(num_samples)
            x2_temp = []
            data_loader = create_data_loader((x1,), batch_size=1000, drop_last=False, shuffle=False)
            for data in data_loader:
                data = data[0]
                x2_logits = self.x2_cond_x1.forward(data)
                x2_prob = F.softmax(x2_logits, dim=-1)
                distribution = Categorical(probs=x2_prob)
                x2 = distribution.sample().cpu().numpy()
                x2_temp.append(x2)
            x2 = np.concatenate(x2_temp, axis=0)
            self.train()
            return x1, x2


class TwoDimensionMADE(nn.Module):
    def __init__(self):
        super(TwoDimensionMADE, self).__init__()
        self.model = MADE(nin=2, hidden_sizes=[32], nout=2 * 200, natural_ordering=True)

    def forward(self, x):
        x = x.type(FloatTensor)
        x = (x - 99.5) / 99.5
        output = self.model.forward(x)
        return output[:, 0::2], output[:, 1::2]

    def sample(self, num_samples):
        self.eval()
        batch_size = 1000
        left_samples = num_samples
        result = []
        while left_samples > 0:
            current_size = min(batch_size, left_samples)
            with torch.no_grad():
                input = np.random.randint(0, 200, (current_size, 2))
                input = torch.from_numpy(input)
                x1_logits, _ = self.forward(input)
                x1_prob = F.softmax(x1_logits, dim=-1)
                distribution = Categorical(probs=x1_prob)
                x1_hat = distribution.sample().cpu().numpy()

                x2 = np.random.randint(0, 200, current_size)
                input = np.stack((x1_hat, x2), axis=-1)
                input = torch.from_numpy(input)
                _, x2_logits = self.forward(input)
                x2_prob = F.softmax(x2_logits, dim=-1)
                distribution = Categorical(probs=x2_prob)
                x2_hat = distribution.sample().cpu().numpy()

                result.append(np.stack((x1_hat, x2_hat), axis=-1))

            left_samples -= current_size
        result = np.concatenate(result, axis=0)
        self.train()
        return result[:, 0], result[:, 1]
