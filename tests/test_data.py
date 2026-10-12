import os
import unittest
import numpy as np
import pandas as pd
from src.utils import load_config
from src.data_loader import generate_synthetic_solar_cell, generate_synthetic_tabular_telemetry
from src.preprocessing import extract_texture_features

class TestDataLoaderAndPreprocessing(unittest.TestCase):

    def test_synthetic_image_generation(self):
        img_healthy = generate_synthetic_solar_cell(class_id=0, size=128)
        img_crack = generate_synthetic_solar_cell(class_id=1, size=128)
        
        self.assertEqual(img_healthy.size, (128, 128))
        self.assertEqual(img_crack.size, (128, 128))

    def test_synthetic_telemetry(self):
        tel = generate_synthetic_tabular_telemetry(class_id=0)
        self.assertIn('Voc', tel)
        self.assertIn('Isc', tel)
        self.assertIn('Fill_Factor', tel)
        self.assertGreater(tel['Voc'], 0.5)

    def test_texture_features(self):
        img_np = np.ones((128, 128, 3), dtype=np.uint8) * 150
        feats = extract_texture_features(img_np)
        self.assertIn('img_mean', feats)
        self.assertIn('glcm_contrast', feats)

if __name__ == '__main__':
    unittest.main()
