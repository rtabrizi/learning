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
            global_crops_scale=(0.4, 1),
            local_crops_scale=(0.05, 0.4),
            n_local_crops=8,
            size=224
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
            [transforms.ToTensor(), transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        self.global_1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=size, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_jitter,
                RandomGaussianBlur(1),  # always apply
                normalize,
            ]
        )

        self.global_2 = transforms.Compose(
            [   # same size local and global for ease of use
                transforms.RandomResizedCrop(
                    size=size, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_jitter,
                RandomGaussianBlur(0.1),
                transforms.RandomSolarize(170, p=0.2),
                normalize,
            ]
        )

        self.local = transforms.Compose([
            transforms.RandomResizedCrop(
                size=size, scale=local_crops_scale, interpolation=Image.BICUBIC),
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

        # weight_g --> magnitude, weight_v --> direction
        # weight_g * (weight_v / weight_v.norm(dim=-1)) we effectively force weights to lie on unit circle
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False))

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

        x = self.mlp(x)  # (num_samples, bottleneck_dim)
        # (num_samples, bottleneck_dim)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)  # (num_samples, output_dim)

        return x


class MultiCropWrapper(nn.Module):
    """For convenience to forward pass the collection of crops

    Parameters
    ----------
    backbone : timm.models.vision_transformer.VisionTransformer
        Instantiated ViT. We will take the 'head' attribute and replace it with 'nn.Identity'

    new_head : Head
        New head that is going to be put on top of the 'backbone'
    """

    def __init__(self, backbone, new_head):
        super().__init__()
        backbone.head = nn.Identity()  # deactivate original head
        self.backbone = backbone
        self.new_head = new_head

    def forward(self, x):
        """Run the forward pass.

        The different crops are concatenated along the batch dimension
        and then a single forward pass is run. The resulting tensor
        is then chunked back to per crop tensors.

        Parameters
        ----------
        x : list
            List of `torch.Tensor` each of shape `(batch_size, 3, size, size)`.

        Returns
        -------
        tuple
            Tuple of `torch.Tensor` each of shape `(batch_size, out_dim)` where
            `output_dim` is determined by `Head`.
        """
        num_crops = len(x)
        # (batch_size * num_crops, 3, size, size)
        concatenated = torch.cat(x, dim=0)
        # (num_samples * num_crops, in_dim)
        cls_embedding = self.backbone(concatenated)
        # (num_samples * num_crops, out_dim)
        logits = self.new_head(cls_embedding)
        chunks = logits.chunk(logits)  # (num_samples * num_crops, out_dim)

        return chunks


class Loss(nn.Module):
    """The loss function.

    We subclass the 'nn.Module' because we want to create a  a buffer for the center logits of the teacher.

    Parameters
    ----------

    out_dim : int
        dimension of the final layer (over which we compute the softmax)

    teacher_temp, student_temp : float
        softmax temperature of the teacher and student floats

    center_momentum : float
        Hyperparameter for the exponential moving average that determines the center logits.
        The highier the momentum, the more the running average matters
    """

    # higher student temp to avoid modal collapse
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """Evaluate loss.

        Parameters
        ----------
        student_output, teacher_output : tuple
        Tuple of tensors of shape (num_samples, out_dim) representing logits.
        Length is equal to the number of crops. Note that the student processed all crops and that the two initial crops
        are the global ones.
        """
        student_temp = [s / self.student_temp for s in student_output]
        teacher_temp = [(t - self.center) /
                        self.teacher_temp for t in teacher_output]

        student_sm = [F.log_softmax(s, dim=-1) for s in student_temp]
        # .detach() since teacher doesn't have gradientes
        teacher_sm = [F.softmax(t, dim=-1).detach() for t in teacher_temp]

        total_loss = 0
        num_loss_terms = 0
        for t_ix, t in enumerate(teacher_sm):
            for s_ix, s in enumerate(student_sm):
                if t_ix == s_ix:
                    continue
                loss = torch.sum(-t * s, dim=-1)  # (num_samples, )
                total_loss += loss.mean()  # scalar
                num_loss_terms += 1
        total_loss /= num_loss_terms
        self.update_center(teacher_output)

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output.

        Compute the exponential moving average.

        Parameters
        ----------
        teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` where each
            tensor represents a different crop.
        """
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True,) # (1, out_dim)
        self.center = 


