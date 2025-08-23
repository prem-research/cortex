"""Tests for Cortex API endpoints"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from server.app.main import app
from server.app.config import get_settings

settings = get_settings()

# Test client
client = TestClient(app)

# Test API key
TEST_API_KEY = "test-key-1"


@pytest.fixture
def auth_headers():
    """Get authorization headers with test API key"""
    return {"Authorization": f"Bearer {TEST_API_KEY}"}


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self):
        """Test health endpoint returns correct status"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "services" in data


class TestAuthEndpoints:
    """Test authentication endpoints"""
    
    def test_generate_token_valid_key(self):
        """Test token generation with valid API key"""
        response = client.post(
            "/auth/token",
            json={"api_key": TEST_API_KEY, "expires_in": 60}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 3600  # 60 minutes in seconds
    
    def test_generate_token_invalid_key(self):
        """Test token generation with invalid API key"""
        response = client.post(
            "/auth/token",
            json={"api_key": "invalid-key", "expires_in": 60}
        )
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]


class TestMemoryEndpoints:
    """Test memory management endpoints"""
    
    def test_store_memory(self, auth_headers):
        """Test storing a new memory"""
        memory_data = {
            "content": "Test memory content",
            "context": "testing",
            "tags": ["test", "example"],
            "user_id": "test_user",
            "session_id": "test_session"
        }
        
        response = client.post(
            "/api/v1/memory",
            json=memory_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "id" in data
        assert data["message"] == "Memory stored successfully"
        
        return data["id"]
    
    def test_search_memories(self, auth_headers):
        """Test searching for memories"""
        # First store a memory
        memory_data = {
            "content": "Python is a great programming language",
            "context": "programming",
            "tags": ["python", "programming"],
            "user_id": "test_user"
        }
        
        store_response = client.post(
            "/api/v1/memory",
            json=memory_data,
            headers=auth_headers
        )
        assert store_response.status_code == 200
        
        # Now search for it
        search_data = {
            "query": "python programming",
            "limit": 5,
            "memory_source": "all",
            "user_id": "test_user"
        }
        
        response = client.post(
            "/api/v1/memory/search",
            json=search_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "memories" in data
        assert "count" in data
        assert isinstance(data["memories"], list)
    
    def test_get_memory_by_id(self, auth_headers):
        """Test getting a specific memory by ID"""
        # First store a memory
        memory_data = {
            "content": "Specific test memory",
            "user_id": "test_user"
        }
        
        store_response = client.post(
            "/api/v1/memory",
            json=memory_data,
            headers=auth_headers
        )
        memory_id = store_response.json()["id"]
        
        # Get the memory
        response = client.get(
            f"/api/v1/memory/{memory_id}",
            params={"user_id": "test_user"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == memory_id
        assert data["content"] == "Specific test memory"
    
    def test_update_memory(self, auth_headers):
        """Test updating a memory"""
        # First store a memory
        memory_data = {
            "content": "Original content",
            "user_id": "test_user"
        }
        
        store_response = client.post(
            "/api/v1/memory",
            json=memory_data,
            headers=auth_headers
        )
        memory_id = store_response.json()["id"]
        
        # Update the memory
        update_data = {
            "memory_id": memory_id,
            "content": "Updated content",
            "tags": ["updated"]
        }
        
        response = client.put(
            "/api/v1/memory",
            json=update_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Memory updated successfully"
    
    def test_delete_memory(self, auth_headers):
        """Test deleting a memory"""
        # First store a memory
        memory_data = {
            "content": "Memory to delete",
            "user_id": "test_user"
        }
        
        store_response = client.post(
            "/api/v1/memory",
            json=memory_data,
            headers=auth_headers
        )
        memory_id = store_response.json()["id"]
        
        # Delete the memory
        delete_data = {
            "memory_id": memory_id,
            "user_id": "test_user"
        }
        
        response = client.delete(
            "/api/v1/memory",
            json=delete_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Memory deleted successfully"
        
        # Verify it's deleted
        get_response = client.get(
            f"/api/v1/memory/{memory_id}",
            params={"user_id": "test_user"},
            headers=auth_headers
        )
        assert get_response.status_code == 404
    
    def test_clear_memories(self, auth_headers):
        """Test clearing memories"""
        # Store some memories first
        for i in range(3):
            memory_data = {
                "content": f"Memory {i} to clear",
                "user_id": "test_clear_user"
            }
            client.post(
                "/api/v1/memory",
                json=memory_data,
                headers=auth_headers
            )
        
        # Clear STM memories
        clear_data = {
            "memory_source": "stm",
            "user_id": "test_clear_user"
        }
        
        response = client.post(
            "/api/v1/memory/clear",
            json=clear_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "cleared from stm" in data["message"].lower()


class TestTemporalSearch:
    """Test temporal and date-based search features"""
    
    def test_temporal_weighted_search(self, auth_headers):
        """Test search with temporal weighting"""
        search_data = {
            "query": "recent updates",
            "limit": 5,
            "memory_source": "all",
            "temporal_weight": 0.7,
            "user_id": "test_user"
        }
        
        response = client.post(
            "/api/v1/memory/search",
            json=search_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "memories" in data
        # Check that temporal weight was applied
        assert "query_metadata" in data
        assert data["query_metadata"]["temporal_weight"] == 0.7
    
    def test_date_range_search(self, auth_headers):
        """Test search with date range filtering"""
        search_data = {
            "query": "test",
            "limit": 5,
            "memory_source": "all",
            "date_range": "last week",
            "user_id": "test_user"
        }
        
        response = client.post(
            "/api/v1/memory/search",
            json=search_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "memories" in data


class TestSystemEndpoints:
    """Test system endpoints"""
    
    def test_get_stats(self, auth_headers):
        """Test getting system statistics"""
        response = client.get(
            "/api/v1/stats",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "total_memories" in data
        assert "stm_capacity" in data
        assert "smart_collections_enabled" in data
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == settings.app_name
        assert data["version"] == settings.app_version
        assert "documentation" in data
        assert "health" in data


class TestErrorHandling:
    """Test error handling"""
    
    def test_unauthorized_access(self):
        """Test accessing protected endpoint without auth"""
        response = client.post(
            "/api/v1/memory",
            json={"content": "test"}
        )
        assert response.status_code == 403  # No auth header provided
    
    def test_invalid_auth_token(self):
        """Test with invalid auth token"""
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.post(
            "/api/v1/memory",
            json={"content": "test"},
            headers=headers
        )
        assert response.status_code == 401
    
    def test_invalid_memory_id(self, auth_headers):
        """Test getting non-existent memory"""
        response = client.get(
            "/api/v1/memory/non-existent-id",
            headers=auth_headers
        )
        assert response.status_code == 404
    
    def test_invalid_request_body(self, auth_headers):
        """Test with invalid request body"""
        response = client.post(
            "/api/v1/memory",
            json={},  # Missing required fields
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])