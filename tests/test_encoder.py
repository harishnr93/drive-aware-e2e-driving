import torch

from driveaware.models.encoders import CompactCNNEncoder
from driveaware.models.encoders.compact_cnn import normalize_images


def test_encoder_shapes_cpu():
    enc = CompactCNNEncoder().eval()
    imgs = torch.randint(0, 256, (2, 96, 160, 3), dtype=torch.uint8)
    x = normalize_images(imgs)
    assert x.shape == (2, 3, 96, 160) and x.dtype == torch.float32
    with torch.no_grad():
        z = enc(x)
    assert z.shape == (2, 256)
    assert torch.isfinite(z).all()


def test_encoder_backprop():
    enc = CompactCNNEncoder()
    z = enc(torch.randn(2, 3, 96, 160))
    z.sum().backward()
    assert all(p.grad is not None for p in enc.parameters())
