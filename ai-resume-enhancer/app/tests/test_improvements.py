"""
Test the /api/v1/improvements/suggest/ endpoint.

This test simulates a user submitting a resume for improvement.
It uses the client fixture to send a POST request to the API and verifies 
that the request is processed successfully and returns a valid response.

Checks:
    - The API returns a 200 OK status
    - The response includes the expected fields:
        original_text, improved_text, suggestions, and action_items
    - Each field is of the correct type 
    - Lists, suggestions and action_items, are not empty 
"""

def test_suggest_improvements(client):
    
    # Input data being sent to API
    payload = {"text": "Experienced software engineer skilled in Python, SQL, and API development."
    }

    # Send POST request with JSON payload
    response = client.post("/api/v1/improvements/suggest/", json=payload)
    
    # Confirm the API processed the request successfully
    assert response.status_code == 200                                      
    
    # JSON to Python dictionary to access contents
    data = response.json()

    # Confirm that all required keys are present
    assert "original_text" in data
    assert "improved_text" in data
    assert "suggestions" in data
    assert "action_items" in data

    # Ensure that each field is of the expected data type
    assert isinstance(data["original_text"], str)
    assert isinstance(data["improved_text"], str)
    assert isinstance(data["suggestions"], list)
    assert isinstance(data["action_items"], list)

    # Confirm that suggestions and action items are not empty
    assert len(data["suggestions"]) > 0
    assert len(data["action_items"]) > 0