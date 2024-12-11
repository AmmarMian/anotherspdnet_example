# FileName: models.py
# Date: 11 Dec 2025 
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: Models used in the project with Kobler
# implementation
# =========================================

from typing import List, Tuple, Optional
import base64

import torch
from torch import nn

# from anotherspdnet.nn import (
#     BiMap, ReEig, LogEig, Vectorization, ReEigBias
# )
# from anotherspdnet.batchnorm import BatchNormSPD

from .modules import BiMap, ReEig, LogEig
from .batchnorm import SPDBatchNorm


class SPDNetKobler(nn.Module):
    def __init__(self,
                 input_dim: int, hidden_layers_size: List[int], output_dim: int,
                 softmax: bool = False, eps: float = 1e-3, 
                 batchnorm: bool = False, seed: Optional[int] = None,
                 precision: Optional[torch.dtype] = torch.float64) -> None:
        """Standard SPDNet model with hidden layers - Kobler Implementation

        Parameters
        ----------
        input_dim : int
            Input dimension of SPDNet

        hidden_layers_size : List[int]
            List of hidden layer sizes

        output_dim : int
            Output dimension of SPDNet

        softmax : bool, optional
            Whether to apply softmax to output, by default False

        eps : float, optional
            Regularization valmue for ReEig, by default 1e-3.

        batchnorm : bool, optional
            Whether to apply batchnorm to hidden layers, by default False

        seed: int, optional
            Random seed

        precision : torch.dtype, optional
            Precision of model, by default torch.float64

        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers_size = hidden_layers_size
        self.output_dim = output_dim
        self.softmax = softmax
        self.eps = eps
        self.batchnorm = batchnorm
        self.seed = seed
        self.precision = precision


        # Initialize seed
        if self.seed is not None:
            torch.seed(self.seed)

        # Create layers
        spdnet_layers = [
                BiMap((1, input_dim, hidden_layers_size[0]), dtype=self.precision)
            ]

        spdnet_layers.append(ReEig(threshold=self.eps))

        if batchnorm:
            spdnet_layers.append(SPDBatchNorm((1, hidden_layers_size[0],
                                               hidden_layers_size[0]),
                                              dtype=self.precision))
        for i in range(1, len(hidden_layers_size)):
            spdnet_layers.append(BiMap((1, hidden_layers_size[i-1],
                                    hidden_layers_size[i]),
                                    dtype=self.precision))
            spdnet_layers.append(ReEig(threshold=self.eps))

            if batchnorm:
                spdnet_layers.append(SPDBatchNorm(hidden_layers_size[i],
                                                dtype=self.precision))
        spdnet_layers.append(LogEig(hidden_layers_size[-1]))

        self.spdnet_layers = nn.Sequential(*spdnet_layers)

        # Create final layer(s)
        self.vectorization = torch.nn.Flatten(start_dim=1)
        self.linear = nn.Linear(
                int(hidden_layers_size[-1]*(hidden_layers_size[-1]+1)/2), output_dim, 
                dtype=self.precision)
                                
        if self.softmax:
            self.softmax_layer = nn.Softmax(dim=-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of SPDNet

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (..., input_dim, input_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., output_dim)
        """
        # Run through SPDNet layers
        X = self.spdnet_layers(X)
        # Run through final layer(s)
        X = self.vectorization(X)
        X = self.linear(X)
        # Apply softmax if required
        if self.softmax:
            X = self.softmax_layer(X)
        return X

    def __repr__(self) -> str:
        """Representation of SPDNet
        Returns
        -------
        str
            String representation of SPDNet
        """
        return f"SPDNet(input_dim={self.input_dim}, "\
                f"hidden_layers_size={self.hidden_layers_size}, "\
                f"output_dim={self.output_dim}, softmax={self.softmax}, "\
                f"eps={self.eps}, seed={self.seed} "\
                f"precision={self.precision}, batchnorm={self.batchnorm}"\

    def __str__(self) -> str:
        """String representation of SPDNet

        Returns
        -------
        str
            String representation of SPDNet
        """
        return self.__repr__()

    def create_model_name_hash(self) -> str:
        """Creates a hash of the model name based on the model parameters

        Returns
        -------
        str
            Hash of model name
        """
        base64_encoded = base64.b64encode(
                self.__str__().encode('utf-8')).decode('utf-8')
        self.model_hash = ''.join(f'{char:02x}' 
                                  for char in base64_encoded.encode('utf-8'))
        return self.model_hash

    def get_model_hash(self) -> str:
        """Returns the model hash

        Returns
        -------
        str
            Model hash
        """
        if not hasattr(self, 'model_hash'):
            self.create_model_name_hash()
        return self.model_hash

    @staticmethod
    def get_model_name_from_hash(hash: str) -> str:
        """Returns the model name from a hash

        Parameters
        ----------
        hash : str
            Hash of model name generated by create_model_name_hash()

        Returns
        -------
        str
            Model name
        """
        decoded_base64 = bytes.fromhex(hash).decode('utf-8')
        decoded_model_name = base64.b64decode(decoded_base64).decode('utf-8')
        return decoded_model_name

    def show_layers(self) -> str:
        """Prints out the layers of SPDNet
        """
        string = self.__repr__()
        string += '\n\nSPDNet Layers:\n'
        string += '---------------\n'
        for i, layer in enumerate(self.spdnet_layers):
            string += f'    ({i}). {layer}\n'
        string += f'    ({len(self.spdnet_layers)+1}). LogEig()\n'
        string += f'    ({len(self.spdnet_layers)+2}). {self.linear}\n'
        if self.softmax:
            string += f'    ({len(self.spdnet_layers)+3}). Softmax()\n'
        print(string)
        return string

    def get_last_tensor(self, X: torch.Tensor) -> torch.Tensor:
        """Returns the last tensor of SPDNet rather than the output of the
        final layer

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (..., input_dim, input_dim)

        Returns
        -------
        torch.Tensor
            Last tensor of SPDNet
        """
        X = self.spdnet_layers(X)
        return X


