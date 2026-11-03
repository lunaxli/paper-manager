from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message":"Hello FastAPI"}

def test_create_paper_invalid_year():

    response = client.post(
        "/papers",
        json={
            "title": "TEST_INVALID",
            "year": 1800
        }
    )

    assert response.status_code == 422

def test_create_paper_invalid_title():
    response = client.post(
        "/papers",
        json={
            "title": "",
            "year":2026
        })
    assert response.status_code == 422