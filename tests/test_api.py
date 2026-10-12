import unittest
from fastapi.testclient import TestClient
from app.api import app

class TestFastAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "online")

    def test_predict_tabular_endpoint(self):
        payload = {
            "Voc": 0.62,
            "Isc": 9.1,
            "Fill_Factor": 79.5,
            "Series_Resistance": 0.015,
            "Shunt_Resistance": 450.0,
            "Temperature_Delta": 1.2,
            "EL_Intensity_Mean": 160.0,
            "EL_Intensity_Std": 20.0
        }
        response = self.client.post("/predict/tabular", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predicted_class_id", data)
        self.assertIn("health_score_index", data)

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("sota_comparison", response.json())

if __name__ == '__main__':
    unittest.main()
