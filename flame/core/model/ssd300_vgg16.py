import warnings
from typing import Any, Optional

from torch import nn
from .ssd import SSD
from .ssd_head import SSDClassificationHead
from .ssd_feature_extractor import _vgg_extractor
from .default_box_generator import DefaultBoxGenerator

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = {
    'ssd300_vgg16_coco': 'https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth',
}


def _validate_trainable_layers(pretrained, trainable_backbone_layers, max_value, default_value):
    # dont freeze any layers if pretrained model or backbone is not used
    if not pretrained:
        if trainable_backbone_layers is not None:
            warnings.warn(
                "Changing trainable_backbone_layers has not effect if "
                "neither pretrained nor pretrained_backbone have been set to True, "
                "falling back to trainable_backbone_layers={} so that all layers are trainable".format(max_value))
        trainable_backbone_layers = max_value

    # by default freeze first blocks
    if trainable_backbone_layers is None:
        trainable_backbone_layers = default_value
    assert 0 <= trainable_backbone_layers <= max_value
    return trainable_backbone_layers


def ssd300_vgg16(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 91,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any
):
    """Constructs an SSD model with input size 300x300 and a VGG16 backbone.
    Reference: `"SSD: Single Shot MultiBox Detector" <https://arxiv.org/abs/1512.02325>`_.
    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes but they will be resized
    to a fixed size before passing it to the backbone.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows, where ``N`` is the number of detections:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each detection
        - scores (Tensor[N]): the scores for each detection
    Example:
        >>> model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 300), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 5)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False

    backbone = _vgg_extractor("vgg16_features", False, progress, pretrained_backbone, trainable_backbone_layers)
    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                           scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                                           steps=[8, 16, 32, 64, 100, 300])

    defaults = {
        # Rescale the input in a way compatible to the backbone
        "image_mean": [0.48235, 0.45882, 0.40784],
        "image_std": [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],  # undo the 0-1 scaling of toTensor
    }
    kwargs = {**defaults, **kwargs}
    model = SSD(backbone, anchor_generator, (300, 300), num_classes, **kwargs)
    if pretrained:
        weights_name = 'ssd300_vgg16_coco'
        if model_urls.get(weights_name, None) is None:
            raise ValueError("No checkpoint is available for model {}".format(weights_name))
        state_dict = load_state_dict_from_url(model_urls[weights_name], progress=progress)
        model.load_state_dict(state_dict)
    return model


class SSD300VGG16(nn.Module):
    def __init__(self, num_classes, pretrained=False, pretrained_backbone=False):
        super(SSD300VGG16, self).__init__()
        self.model = self.ssd300_vgg16(num_classes, pretrained, pretrained_backbone)

    def ssd300_vgg16(self, num_classes, pretrained, pretrained_backbone):
        model = ssd300_vgg16(pretrained=pretrained, pretrained_backbone=pretrained_backbone)
        out_channels = model.out_channels
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDClassificationHead(in_channels=out_channels,
                                                               num_anchors=num_anchors,
                                                               num_classes=num_classes)
        return model

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, x, targets=None):
        return self.model(x, targets)
