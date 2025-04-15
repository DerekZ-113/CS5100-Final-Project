from fastapi.testclient import TestClient
from app.main import app

# Create a test client
client = TestClient(app)

def test_suggest_improvements():
    payload = {
        "text": "Experienced Python developer with knowledge of REST APIs and cloud computing."
    }

    response = client.post("/api/v1/improvements/suggest/", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "original_text" in data
    assert "improved_text" in data
    assert "suggestions" in data
    assert isinstance(data["suggestions"], list)