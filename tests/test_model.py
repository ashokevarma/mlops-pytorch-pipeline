"""Unit tests for the model factory."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import SimpleCNN, get_model  # noqa: E402


@pytest.mark.parametrize("architecture", ["resnet18", "simple_cnn"])
def test_forward_output_shape(architecture):
    num_classes = 10
    model = get_model(architecture=architecture, num_classes=num_classes)
    model.eval()
    dummy = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (4, num_classes)


def test_custom_num_classes():
    model = get_model(architecture="simple_cnn", num_classes=3)
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 3)


def test_simple_cnn_is_returned():
    assert isinstance(get_model("simple_cnn"), SimpleCNN)


def test_unknown_architecture_raises():
    with pytest.raises(ValueError):
        get_model(architecture="does-not-exist")
