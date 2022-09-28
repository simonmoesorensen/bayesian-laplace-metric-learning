from src.baselines.models import CIFAR10ConvNet, CIFAR10LinearNet, FashionMNISTConvNet, FashionMNISTLinearNet, CUB200ConvNet, SampleNet


class MCDropoutHead(SampleNet):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def pass_through(self, x):
        x = self.backbone(x)
        return x


def MNIST_MCDropout(embedding_size=128, linear=False):
    """
    Construct a mnist model for MCDropout.
    """
    if linear:
        model = FashionMNISTLinearNet(latent_dim=embedding_size, p=0.1)
    else:
        model = FashionMNISTConvNet(latent_dim=embedding_size, p=0.1)
    model_mc = MCDropoutHead(model)

    return model_mc


def CIFAR10_MCDropout(embedding_size=128, linear=False):
    """
    Construct a cifar10 model for MCDropout.
    """
    if linear:
        model = CIFAR10LinearNet(latent_dim=embedding_size, p=0.1)
    else:
        model = CIFAR10ConvNet(latent_dim=embedding_size, p=0.1)
    model_mc = MCDropoutHead(model)

    return model_mc

def CUB200_MCDropout(embedding_size=128, linear=False):
    """
    Construct a cifar10 model for MCDropout.
    """
    if linear:
        raise NotImplementedError
    model = CUB200ConvNet(latent_dim=embedding_size, p=0.1)
    model_mc = MCDropoutHead(model)

    return model_mc

