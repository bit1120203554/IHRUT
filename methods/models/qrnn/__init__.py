from .qrnn3d import QRNNREDC3D
from .redc3d import REDC3D
from .resnet import ResQRNN3D


def get_qrnn3d():
    net = QRNNREDC3D(1, 16, 5, [1, 3], has_ad=True)
    net.use_2dconv = False
    net.bandwise = False
    return net
