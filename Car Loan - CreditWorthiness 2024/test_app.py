import pytest # type: ignore
from app import app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_ping(client):
    response = client.get('/ping')
    assert response.status_code == 200
    assert response.json == {"message": "Pinging the NEWWWW model successful!!"}


def test_predict_approved(client):
    test_input = {
        "Occupation": "Teacher",
        "Monthly Income": 6000,
        "Credit Score": 800,
        "Years of Employment": 10,
        "Finance Status": "Excellent",
        "Finance History": "Good",
        "Number of Children": 2
    }
    response = client.post('/predict', json=test_input)
    assert response.status_code == 200
    assert response.json == {"prediction": "Approved"}


# Add more test cases as needed
