from src.baselines.models import CIFAR10ConvNet, FashionMNISTConvNet, SampleNet
from src.utils import L2Norm


class MCDropoutHead(SampleNet):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.norm = L2Norm()

    def pass_through(self, x):
        x = self.backbone(x)
        return self.norm(x)


def MNIST_MCDropout(embedding_size=128):
    """
    Construct a mnist model for MCDropout.
    """
    model = FashionMNISTConvNet(latent_dim=embedding_size)
    model_mc = MCDropoutHead(model)

    return model_mc


def CIFAR10_MCDropout(embedding_size=128):
    """
    Construct a cifar10 model for MCDropout.
    """
    model = CIFAR10ConvNet(latent_dim=embedding_size)
    model_mc = MCDropoutHead(model)

    return model_mc
