import unittest
import torch
import numpy as np
from src.models import SolarResNetClassifier, SolarCustomCNN

class TestModels(unittest.TestCase):

    def test_resnet_forward_pass(self):
        model = SolarResNetClassifier(num_classes=4, backbone="resnet18", pretrained=False)
        dummy_input = torch.randn(2, 3, 128, 128)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 4))

    def test_custom_cnn_forward_pass(self):
        model = SolarCustomCNN(num_classes=4)
        dummy_input = torch.randn(2, 3, 128, 128)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 4))

if __name__ == '__main__':
    unittest.main()
