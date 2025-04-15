"""
Test for the /api/health endpoint.

This test ensures that the API has successfully started, is responsive, and returns the expected health status.
"""


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200                                          # Assert the server responds with 200 OK
    assert response.json() == {"status": "healthy", "version": "1.0.0"}         # Assert the JSON response matches the expected structure and values