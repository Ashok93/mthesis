import torch.utils.model_zoo as model_zoo
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, model_urls


class ResnetFeatureExtractor(ResNet):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResnetFeatureExtractor, self).__init__(block, layers, num_classes, zero_init_residual,
                                                     groups, width_per_group, replace_stride_with_dilation,
                                                     norm_layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


def resnet18_feature_extractor(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model as a feature extractor.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResnetFeatureExtractor(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
