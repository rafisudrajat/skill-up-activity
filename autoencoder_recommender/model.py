import torch
from torch import nn

class AutoRecItemBased(nn.Module):
    def __init__(self, num_hidden: int, num_users: int, dropout:float=0.05):
        """
        AutoRec item based model for collaborative filtering.

        Parameters:
        -----------
        num_hidden : int
            The number of hidden units in the encoder.
        num_users : int
            The number of users (also the number of output units in the decoder).
        dropout : float, optional
            The dropout rate. Default is 0.05.
        """
        super(AutoRecItemBased, self).__init__()
        self.encoder = nn.Linear(num_users,num_hidden)
        self.decoder = nn.Linear(num_hidden, num_users)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AutoRec model.

        Parameters:
        -----------
        input : torch.Tensor
            The input tensor with shape (batch_size, num_users).

        Returns:
        --------
        torch.Tensor
            The output tensor with shape (batch_size, num_users).
        """
        hidden = self.dropout(torch.relu(self.encoder(input)))
        pred = self.decoder(hidden)
        return pred
        