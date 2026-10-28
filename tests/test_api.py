
import math

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


# Example request based on the used-car dataset.
# We will verify these fields against model.joblib before running.
VALID_CAR = {
    "brand": "Toyota",
    "model": "Camry",
    "model_year": 2022,
    "milage": 25000,
    "fuel_type": "Gasoline",
    "engine": "2.5L I4",
    "transmission": "Automatic",
    "ext_col": "White",
    "int_col": "Black",
    "accident": "None reported",
    "clean_title": "Yes",
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_returns_success():
    response = client.post("/predict", json=VALID_CAR)

    assert response.status_code == 200


def test_prediction_contains_price():
    response = client.post("/predict", json=VALID_CAR)

    assert response.status_code == 200

    result = response.json()
    assert "predicted_price" in result

    price = result["predicted_price"]
    assert isinstance(price, (int, float))
    assert math.isfinite(price)
    assert price > 0


def test_prediction_rejects_missing_fields():
    response = client.post("/predict", json={})

    assert response.status_code == 422
