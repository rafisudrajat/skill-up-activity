
import torch
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, num_factors: int, num_users: int, num_items: int):
        """
        Matrix Factorization model using PyTorch.

        Parameters:
        -----------
        num_factors : int
            The number of latent factors.
        num_users : int
            The number of unique users.
        num_items : int
            The number of unique items.
        """
        super(MF, self).__init__()
        # User latent matrix
        self.P = nn.Embedding(num_users, num_factors)
        # Item latent matrix
        self.Q = nn.Embedding(num_items, num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id: torch.Tensor, item_id: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.

        Parameters:
        -----------
        user_id : torch.Tensor
            Tensor containing user indices.
        item_id : torch.Tensor
            Tensor containing item indices.

        Returns:
        --------
        torch.Tensor
            Predicted ratings.
        """
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id).squeeze()
        b_i = self.item_bias(item_id).squeeze()
        outputs = (P_u * Q_i).sum(dim=1) + b_u + b_i
        return outputs
    