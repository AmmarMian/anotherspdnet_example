# FileName: models.py
# Date: 29 juin 2023 - 18:13
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: Models used in the project
# =========================================

from typing import List, Tuple, Optional
import base64

import torch
from torch import nn

from anotherspdnet.nn import (
    BiMap, ReEig, LogEig, Vectorization, ReEigBias
)
from anotherspdnet.batchnorm import BatchNormSPD

from .utils import is_spd

class SPDNet(nn.Module):
    def __init__(self,
                 input_dim: int, hidden_layers_size: List[int], output_dim: int,
                 softmax: bool = False, eps: float = 1e-3, 
                 reeig_bias: bool = False,
                 batchnorm: bool = False, seed: Optional[int] = None,
                 device: Optional[torch.device] = None,
                 precision: Optional[torch.dtype] = torch.float64,
                 max_iter_batchnorm: int = 5) -> None:
        """Standard SPDNet model with hidden layers

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

        reeig_bias : bool, optional
            Whether to use ReEigBias instead of ReEig, by default False

        batchnorm : bool, optional
            Whether to apply batchnorm to hidden layers, by default False

        seed : int, optional
            Random seed for layers initialization, by default None

        device : torch.device, optional
            Device to run model on, by default None = torch.device('cpu')

        precision : torch.dtype, optional
            Precision of model, by default torch.float64

        max_iter_batchnorm : int, optional
            Maximum number of iterations for mean computation batchnorm, by default 5.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers_size = hidden_layers_size
        self.output_dim = output_dim
        self.softmax = softmax
        self.eps = eps
        self.seed = seed
        self.device = device if device is not None else torch.device('cpu')
        self.precision = precision
        self.batchnorm = batchnorm
        self.max_iter_batchnorm = max_iter_batchnorm
        self.reeig_bias = reeig_bias


        # Initialize seed
        if self.seed is not None:
            rng = torch.manual_seed(self.seed)

        # Create layers
        spdnet_layers = [
                BiMap(input_dim, hidden_layers_size[0], seed=self.seed,
                    device=self.device, dtype=self.precision),
            ]

        if reeig_bias:
            spdnet_layers.append(ReEigBias(eps=eps, dim=hidden_layers_size[0],
                                        device=self.device,
                                        dtype=self.precision))
        else:
            spdnet_layers.append(ReEig(eps=eps, dim=hidden_layers_size[0]))

        if batchnorm:
            spdnet_layers.append(BatchNormSPD(hidden_layers_size[0],
                                              device=self.device,
                                              dtype=self.precision,
                                              max_iter_mean=max_iter_batchnorm))
        for i in range(1, len(hidden_layers_size)):
            spdnet_layers.append(BiMap(hidden_layers_size[i-1],
                                    hidden_layers_size[i],
                                    seed=self.seed,
                                    device=self.device,
                                    dtype=self.precision))
            if reeig_bias:
                spdnet_layers.append(ReEigBias(eps=eps,
                                dim=hidden_layers_size[i],
                                device=self.device,
                                dtype=self.precision))
            else:
                spdnet_layers.append(ReEig(eps=eps, dim=hidden_layers_size[i]))

            if batchnorm:
                spdnet_layers.append(BatchNormSPD(hidden_layers_size[i],
                                                device=self.device,
                                                dtype=self.precision,
                                                max_iter_mean=max_iter_batchnorm))
        spdnet_layers.append(LogEig())

        self.spdnet_layers = nn.Sequential(*spdnet_layers)

        # Create final layer(s)
        self.vectorization = Vectorization()
        self.linear = nn.Linear(hidden_layers_size[-1]**2, output_dim, 
                                dtype=self.precision, device=self.device)
                                
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
                f"eps={self.eps}, seed={self.seed}, device={self.device}, "\
                f"precision={self.precision}, batchnorm={self.batchnorm}, "\
                f"max_iter_batchnorm={self.max_iter_batchnorm}, "\
                f"reeig_bias={self.reeig_bias})"

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


# Encoder-Decoder models
# ======================
class SPDNetAutoEncoder(nn.Module):
    """Autoencoder for SPD matrices"""

    def __init__(self,
                hd: List[int],
                n_classes: int,
                eps: float = 1e-3,
                reeig_bias: bool = False, softmax: bool = False,
                batchnorm: bool = False, seed: Optional[int] = None,
                device: Optional[torch.device] = None,
                precision: Optional[torch.dtype] = torch.float64,
                max_iter_batchnorm: int = 5) -> None:
        super().__init__()
        self.hd = hd
        self.eps = eps
        self.seed = seed
        self.device = device if device is not None else torch.device('cpu')
        self.precision = precision
        self.batchnorm = batchnorm
        self.max_iter_batchnorm = max_iter_batchnorm
        self.reeig_bias = reeig_bias
        self.n_classes = n_classes
        self.softmax = softmax

        # Initialize seed
        if self.seed is not None:
            rng = torch.manual_seed(self.seed)

        # Encoder layers
        encoder_layers = [
                BiMap(n_classes, hd[0], seed=self.seed,
                    device=self.device, dtype=self.precision),
                            ]
        if reeig_bias:
            encoder_layers.append(ReEigBias(eps=eps, dim=hd[0],
                                device=self.device,
                                dtype=self.precision))
        if batchnorm:
            encoder_layers.append(BatchNormSPD(hd[0],
                                        device=self.device,
                                        dtype=self.precision,
                                        max_iter_mean=max_iter_batchnorm))
        for i in range(1, len(hd)):
            encoder_layers.append(BiMap(hd[i-1], hd[i], seed=self.seed,
                            device=self.device, dtype=self.precision))
            if reeig_bias:
                encoder_layers.append(ReEigBias(eps=eps, dim=hd[i],
                            device=self.device,
                            dtype=self.precision))
            if batchnorm:
                encoder_layers.append(BatchNormSPD(hd[i],
                                device=self.device,
                                dtype=self.precision,
                                max_iter_mean=max_iter_batchnorm))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers
        reversed_hd = hd[::-1]
        decoder_layers = [
                BiMap(reversed_hd[0], n_classes, seed=self.seed,
                    device=self.device, dtype=self.precision),
                            ]
        if reeig_bias:
            decoder_layers.append(ReEigBias(eps=eps, dim=n_classes,
                        device=self.device,
                        dtype=self.precision))
        if batchnorm:
            decoder_layers.append(BatchNormSPD(n_classes,
                                device=self.device,
                                dtype=self.precision,
                                max_iter_mean=max_iter_batchnorm))
        for i in range(1, len(reversed_hd)):
            decoder_layers.append(BiMap(reversed_hd[i-1], reversed_hd[i],
                    seed=self.seed, device=self.device,
                    dtype=self.precision))
            if reeig_bias:
                decoder_layers.append(ReEigBias(eps=eps, dim=reversed_hd[i],
                    device=self.device,
                    dtype=self.precision))
            if batchnorm:
                decoder_layers.append(BatchNormSPD(reversed_hd[i],
                    device=self.device,
                    dtype=self.precision,
                    max_iter_mean=max_iter_batchnorm))

        self.decoder = nn.Sequential(*decoder_layers)

        # Classification layers
        classification = [LogEig(), Vectorization(),
                          nn.Linear(hd[-1]**2, n_classes)]
        if softmax:
            classification.append(nn.Softmax(dim=-1))
        self.classification = nn.Sequential(*classification)

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """Encode SPD matrices

        Parameters
        ----------
        X : torch.Tensor
            Input SPD matrices

        Returns
        -------
        torch.Tensor
            Encoded SPD matrices
        """
        return self.encoder(X)

    def decode(self, X: torch.Tensor) -> torch.Tensor:
        """Decode SPD matrices
        Parameters
        ----------
        X : torch.Tensor
            Input SPD matrices
        Returns
        -------
        torch.Tensor
            Decoded SPD matrices
        """
        return self.decoder(X)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of SPDNetAutoEncoder
        Parameters
        ----------
        X : torch.Tensor
            Input SPD matrices

        Returns
        -------
        torch.Tensor
            Output SPD matrices
        """
        encoding = self.encode(X)
        classification = self.classification(encoding)
        return self.decode(encoding), classification


    def __repr__(self) -> str:
        """Representation of SPDNetAutoEncoder

        Returns
        -------
        str
            String representation of SPDNetAutoEncoder
        """
        return f"SPDNetAutoEncoder(hd={self.hd}, eps={self.eps}, "\
            f"seed={self.seed}, device={self.device}, "\
            f"softmax={self.softmax}, "\
            f"precision={self.precision}, batchnorm={self.batchnorm}, "\
            f"max_iter_batchnorm={self.max_iter_batchnorm}, "\
            f"reeig_bias={self.reeig_bias})"

    def __str__(self) -> str:
        """String representation of SPDNetAutoEncoder

        Returns
        -------
        str
            String representation of SPDNetAutoEncoder
        """
        return self.__repr__()

