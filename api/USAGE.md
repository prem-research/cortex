# Cortex Memory API - Usage Guide

A comprehensive guide for testing and using the Cortex Memory API with detailed examples and best practices.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Authentication Methods](#authentication-methods)
- [Complete Testing Workflow](#complete-testing-workflow)
- [API Endpoint Reference](#api-endpoint-reference)
- [Advanced Usage Patterns](#advanced-usage-patterns)
- [Response Formats](#response-formats)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Prerequisites

1. **Service Running**: Ensure all services are running with `make up`
2. **Health Check**: Verify with `make health` that all services are healthy
3. **Tools**: Have `curl` and `jq` (optional, for JSON formatting) available

## Authentication Method

The API uses JWT Bearer token authentication:

### JWT Tokens
- Login to get a JWT token
- Include in `Authorization: Bearer <token>` header
- Tokens are valid for 365 days

## Complete Testing Workflow

### Step 1: Health Verification
```bash
# Check API health
curl http://localhost:7001/api/v1/health/

# Check Cortex system health (tests ChromaDB connectivity)
curl http://localhost:7001/api/v1/health/cortex/
```

**Expected Response:**
```json
{
  "status": "healthy",
  "service": "cortex-api",
  "timestamp": "2024-01-01T10:00:00Z"
}
```

### Step 2: User Registration
```bash
# Register a new user
curl -X POST "http://localhost:7001/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "demo_user",
    "email": "demo@example.com",
    "password": "demo123"
  }'
```

**Expected Response:**
```json
{
  "id": 1,
  "username": "demo_user",
  "email": "demo@example.com",
  "is_active": true
}
```

### Step 3: Authentication Testing
```bash
# Login to get JWT token
curl -X POST "http://localhost:7001/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "demo_user",
    "password": "demo123"
  }'
```

**Expected Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user_id": "1"
}
```

```bash
# Save the JWT token for subsequent requests
export JWT_TOKEN="your-jwt-token-here"

# Test token validation
curl -X GET "http://localhost:7001/api/v1/auth/me" \
  -H "Authorization: Bearer $JWT_TOKEN"
```

### Step 4: Adding Memories

```bash
# Add a work-related memory
curl -X POST "http://localhost:7001/api/v1/memory/add" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Had a productive meeting with the development team about Q1 goals and roadmap planning. Discussed new features, timeline adjustments, and resource allocation.",
    "metadata": {
      "priority": "high",
      "category": "meeting",
      "attendees": 8,
      "duration": "2 hours"
    }
  }'

# Add a personal memory
curl -X POST "http://localhost:7001/api/v1/memory/add" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Learned a new recipe for homemade pasta carbonara. Key ingredients: eggs, pecorino cheese, guanciale, and black pepper. The secret is to remove from heat when adding eggs.",
    "metadata": {
      "category": "cooking",
      "difficulty": "medium",
      "prep_time": "30 minutes",
      "cuisine": "Italian"
    }
  }'

# Add a technical memory
curl -X POST "http://localhost:7001/api/v1/memory/add" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Fixed a critical bug in the authentication service. The issue was related to JWT token expiration handling in the middleware. Solution involved updating the token refresh logic.",
    "metadata": {
      "priority": "critical",
      "category": "bug_fix",
      "component": "auth",
      "time_spent": "4 hours",
      "status": "resolved"
    }
  }'

# Add a learning memory
curl -X POST "http://localhost:7001/api/v1/memory/add" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Read an excellent article about vector databases and their applications in AI. ChromaDB seems to be gaining popularity for embeddings storage. Key points: similarity search, metadata filtering, and persistence.",
    "metadata": {
      "category": "research",
      "topic": "ai",
      "source": "article",
      "url": "https://example.com/vector-db-article",
      "reading_time": "15 minutes"
    }
  }'
```

> **Note**: You can organize memories using `session_id` for different contexts (e.g., `"session_id": "work_project"`, `"session_id": "personal"`, `"session_id": "learning"`). If omitted, memories are stored in the default session. But Cortex's Smart Auto Collections should be able to handle this internally to maintain good segregation.

### Step 5: Memory Search Testing

```bash
# Basic search
curl -X POST "http://localhost:7001/api/v1/memory/search" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "team meeting goals",
    "limit": 5
  }'

# Search with temporal weighting (favor recent memories)
curl -X POST "http://localhost:7001/api/v1/memory/search" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "cooking recipe",
    "temporal_weight": 0.8,
    "limit": 5
  }'

# Search with metadata filtering
curl -X POST "http://localhost:7001/api/v1/memory/search" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "development",
    "where_filter": {"priority": {"$eq": "high"}},
    "limit": 5
  }'

# Search with date range
curl -X POST "http://localhost:7001/api/v1/memory/search" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "bug fix",
    "date_range": "today",
    "limit": 10
  }'

# Search with specific date range (RFC3339 format)
curl -X POST "http://localhost:7001/api/v1/memory/search" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "project updates",
    "date_range": "{\"start\": \"2024-01-01T00:00:00Z\", \"end\": \"2024-12-31T23:59:59Z\"}",
    "limit": 10
  }'
```

### Step 6: Memory Retrieval

```bash
# Get specific memory by ID (replace with actual memory ID from search results)
curl -X GET "http://localhost:7001/api/v1/memory/get/550e8400-e29b-41d4-a716-446655440000" \
  -H "Authorization: Bearer $JWT_TOKEN"

# Get linked memories for specific memory IDs
curl -X POST "http://localhost:7001/api/v1/memory/get-linked" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_ids": ["550e8400-e29b-41d4-a716-446655440000", "550e8400-e29b-41d4-a716-446655440001"],
    "limit": 4
  }'

# Get memories with their linked context
curl -X POST "http://localhost:7001/api/v1/memory/get-with-linked" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_ids": ["550e8400-e29b-41d4-a716-446655440000", "550e8400-e29b-41d4-a716-446655440001"],
    "linked_limit": 3
  }'
```

### Step 7: Memory Management

```bash
# Clear memories (use with caution - this will delete data!)
curl -X POST "http://localhost:7001/api/v1/memory/clear" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": ""
  }'
```

## API Endpoint Reference

### Authentication Endpoints
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login and get JWT token
- `GET /api/v1/auth/me` - Get current user info

### Memory Management Endpoints
- `POST /api/v1/memory/add` - Add new memory
- `POST /api/v1/memory/search` - Search memories
- `GET /api/v1/memory/get/{memory_id}` - Get specific memory
- `POST /api/v1/memory/get-linked` - Get linked memories
- `POST /api/v1/memory/get-with-linked` - Get memories with context
- `POST /api/v1/memory/clear` - Clear memories

### Health Endpoints
- `GET /api/v1/health` - API health status
- `GET /api/v1/health/cortex` - Cortex system health

## Advanced Usage Patterns

### Complex Metadata Filtering
```bash
curl -X POST "http://localhost:7001/api/v1/memory/search" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "development work",
    "where_filter": {
      "$and": [
        {"category": {"$eq": "meeting"}},
        {"priority": {"$in": ["high", "critical"]}},
        {"attendees": {"$gte": 5}}
      ]
    },
    "limit": 5
  }'
```

### Session-Based Memory Organization
```bash
# Add memory to specific session
curl -X POST "http://localhost:7001/api/v1/memory/add" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Project Alpha milestone completed successfully.",
    "session_id": "project_alpha",
    "metadata": {"milestone": "M1", "status": "completed"}
  }'

# Search within specific session
curl -X POST "http://localhost:7001/api/v1/memory/search" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "milestone",
    "session_id": "project_alpha",
    "limit": 10
  }'
```

## Response Formats

### Successful Memory Addition
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Memory added successfully",
  "user_id": 1,
  "session_id": ""
}
```

### Search Results
```json
{
  "memories": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "content": "Had a productive meeting with the development team...",
      "metadata": {
        "priority": "high",
        "category": "meeting",
        "attendees": 8
      },
      "timestamp": "2024-01-01T10:00:00Z",
      "similarity_score": 0.87
    }
  ],
  "total_results": 1,
  "query_processed": "team meeting development productive"
}
```

### Error Responses
```json
{
  "detail": "Invalid authentication credentials"
}
```

```json
{
  "detail": [
    {
      "loc": ["body", "content"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

## Best Practices

### 1. Session Organization
- Use meaningful session IDs: `work_project_alpha`, `personal_2024`, `learning_ai`
- Keep sessions focused on specific contexts or time periods
- Use empty session (`""`) for general memories

### 2. Metadata Strategy
- Include relevant metadata for better filtering and organization
- Use consistent key names across similar memory types
- Include temporal metadata like `created_date`, `deadline`, `duration`

### 3. Search Optimization
- Use specific queries for better semantic matching
- Combine temporal weighting with metadata filtering for precise results
- Use appropriate limits to balance performance and completeness

### 4. Authentication Security
- Store JWT tokens securely and refresh before expiration (tokens valid for 365 days)
- Never expose authentication credentials in logs or client-side code

## Troubleshooting

### Common Issues

#### 401 Unauthorized
**Cause**: Invalid or expired JWT token
**Solution**: Re-login to get fresh token

#### 422 Validation Error
**Cause**: Malformed request body or missing required fields
**Solution**: Check request format against API documentation

#### 500 Internal Server Error
**Cause**: Service connectivity issues, database problems
**Solution**: Check service health with `make health` and logs with `make logs`

#### Empty Search Results
**Cause**: No matching memories, overly restrictive filters
**Solution**: Try broader search terms, remove filters, check session_id

### Debugging Commands

```bash
# Check all services status
make status

# View live logs
make logs

# Health check all services
make health

# Restart services if needed
make restart
```

---

**Next Steps:**
- Explore the [API Documentation](http://localhost:7001/docs) for complete endpoint details
- Check out the [README](README.md) for deployment and configuration options
- Review logs with `make logs` if you encounter any issues