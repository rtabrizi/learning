# original paper trained on ImageNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

class DataAugmentation:
    """Create crops of an input image together with additional augmentations.
    
    It generates 2 global crops and 'n_local_crops' local crops.

    Parameters
    ----------
    global_crops_scale : tuple
        Range of sizes for the global crops.

    local_crops_scale : tuple
        Range of sizes for the local crops.
    
    n_local_crops : int
        Number of local crops to create.

    size : int
        size of the final image.

    Attributes
    ----------
    global_1, global_2 : transforms.Compose
        Two global transforms.

    local : transforms.Compose
        local transform. Note that the augmentation is stochastic so one instance is enough
        and will lead to different crops
    """
    def __init__(
            self,
            global_crops_scale = (0.4, 1),
            local_crops_scale = (0.05, 0.4),
            n_local_crops = 8,
            size = 224
    ):
        self.n_local_crops = n_local_crops
        RandomGaussianBlur = lambda p: transforms.RandomApply([
            transforms.GaussianBlur(kernel_zize=5, sigma=(0.1, 2.0))
        ])

        # random horizontal flips, grayscale, and jitter
        flip_and_jitter = transforms.compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitterb(rightness=0.4, contrast=0.4, saturation=0.2, hue=0.2)]),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        normalize = transforms.Compose( 
            # imagenet mean and std across all 3 color channels
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])]
        )

        self.global_1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_jitter,
                RandomGaussianBlur(1), # always apply
                normalize,
            ]
        )

        self.global_2 = transforms.Compose(
            [   # same size local and global for ease of use
                transforms.RandomResizedCrop(size=size, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_jitter,
                RandomGaussianBlur(0.1),
                transforms.RandomSolarize(170, p=0.2),
                normalize,
            ]
        )
        
        self.local = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_jitter,
            RandomGaussianBlur(0.5),
            normalize
        ]
        )

    def __call__(self, img):
        """Apply the augmentations to an input image."""

        all_crops = []
        all_crops.append(self.global_1(img))
        all_crops.append(self.global_2(img))

        all_crops.extend([self.local(img) for _ in range(self.n_local_crops)])


        return all_crops
    
class Head(nn.Module):
    """Network hooked up to the CLS token embedding.
    
    MLP with last layer being normalized in a particular way
    
    Parameters
    ----------

    in_dim : int
        Token embedding dimension

    out_dim : int
        Dimension of final layer (which we compute the softmax over)

    hidden_dim : int
        Dim of hidden layers

    bottleneck_dim : int
        Dim of second to last layer

    num_layers : int
        Number of hidden layers
    
    norm_last_layer : bool
        If True, freeze norm of the weight of the last linear layer to 1
    
    Attributes
    ----------
    mlp: nn.sequential
        vanilla MLP
    
        last_layer : nn.Linear
            Last layer of the MLP with weight normalization. We'll use 'weight_g' and 'weight_v" as 
            learnable parameters instead of a single 'weight.'
    """
    def __init__(self, in_dim=384, out_dim=1000, hidden_dim=512, bottleneck_dim=256, num_layers=3, norm_last_layer=False):
        super().__init__()
        if num_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.sequential(*layers)
        
        self.apply(self._init_weights)

        #weight_g --> magnitude, weight_v --> direction
        # weight_g * (weight_v / weight_v.norm(dim=-1)) we effectively force weights to lie on unit circle
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))

        self.last_layer.weight_g_data.fill(1)
        if norm_last_layer:
            # magnitude stays 
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Run forward pass.
        
        Parameters
        ----------
        
        x : torch.Tesnor
            CLS token of shape (batch_size, in_dim)
            
        Returns
        -------
        torch.Tensor
            Of shape (num_samples, out_dim)    
        """

        x = self.mlp(x) # (num_samples, bottleneck_dim)
        x = nn.functional.normalize(x, dim=-1, p=2) # (num_samples, bottleneck_dim)
        x = self.last_layer(x) # (num_samples, output_dim)
        
        return x