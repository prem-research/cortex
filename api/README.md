# Cortex Memory API

A production-ready API service for the Cortex memory system with multi-user authentication and session support.

## Features

- **Dual-tier Memory**: Fast STM + persistent LTM with ChromaDB
- **Smart Collections**: Context-aware categorization and domain organization
- **Temporal Awareness**: Time-sensitive search with recency weighting
- **Memory Evolution**: Automatic relationship building between memories
- **Multi-user Support**: Complete isolation by user_id and session_id
- **JWT Bearer Authentication**: Secure token-based authentication
- **Docker Deployment**: Production-ready containerization

## Quick Start

### 1. Prerequisites

- Docker and Docker Compose
- OpenAI API key

### 2. Setup

```bash
# Clone and navigate to API directory
cd cortex/api

# Copy environment configuration
cp .env.example .env

# Edit .env file with your settings
OPENAI_API_KEY=sk-your-key-here
JWT_SECRET_KEY=your-secret-key-here
```

### 3. Start Services

```bash
# Build and start all services
make build

# Start services
make up

# Check service health
make health
```

### 4. API Documentation

Visit http://localhost:7001/docs for interactive API documentation.

## Usage & Testing

For detailed usage examples and testing commands, see [USAGE.md](USAGE.md).

## API Usage

### Authentication

#### Register a new user:
```bash
curl -X POST "http://localhost:7001/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "email": "john@example.com", 
    "password": "secure_password123"
  }'
```

#### Login to get JWT token:
```bash
curl -X POST "http://localhost:7001/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "password": "secure_password123"
  }'
```

### Memory Operations

#### Add a memory:
```bash
curl -X POST "http://localhost:7001/api/v1/memory/add" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "I had a productive meeting with the team about Q1 goals",
    "session_id": "work_session",
    "metadata": {"priority": "high"}
  }'
```

#### Search memories:
```bash
curl -X POST "http://localhost:7001/api/v1/memory/search" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "team meeting goals",
    "session_id": "work_session",
    "temporal_weight": 0.3,
    "limit": 5
  }'
```

#### Search with date range:
```bash
curl -X POST "http://localhost:7001/api/v1/memory/search" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "project updates",
    "date_range": "{\"start\": \"2024-01-01T00:00:00Z\", \"end\": \"2024-01-31T23:59:59Z\"}",
    "limit": 10
  }'
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:postgres@localhost:7432/cortex_api` |
| `CHROMA_URI` | ChromaDB server URL | `http://localhost:7003` |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | **Required** |
| `JWT_SECRET_KEY` | Secret key for JWT tokens | **Required** |

### Docker Ports

- **7001**: Cortex API server
- **7003**: ChromaDB vector database  
- **7432**: PostgreSQL database

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login and get JWT token
- `GET /api/v1/auth/me` - Get current user info

### Memory Management
- `POST /api/v1/memory/add` - Add new memory
- `POST /api/v1/memory/search` - Search memories
- `GET /api/v1/memory/get/{memory_id}` - Get specific memory
- `POST /api/v1/memory/get-linked` - Get linked memories
- `POST /api/v1/memory/get-with-linked` - Get memories with context
- `POST /api/v1/memory/clear` - Clear memories

### Health Checks
- `GET /api/v1/health` - API health status
- `GET /api/v1/health/cortex` - Cortex system health

## Memory Organization

### User Isolation
- Each user has completely isolated memory space
- `user_id` is the primary identifier (automatically set from authentication)
- `session_id` is optional for further organization within a user's memories

### Session Management
- Empty `session_id` (''): Default session for the user
- Custom `session_id`: Organize memories by conversation, project, etc.
- Sessions are user-scoped (User A's "work" session ≠ User B's "work" session)

### Advanced Search Features

#### Temporal Weighting
Balance semantic similarity with recency:
```json
{
  "query": "project updates",
  "temporal_weight": 0.7,  // 70% recency, 30% semantic
  "limit": 10
}
```

#### Date Range Filtering
Filter by specific time periods:
```json
{
  "query": "team meetings",
  "date_range": "{\"start\": \"2024-01-01T00:00:00Z\", \"end\": \"2024-01-31T23:59:59Z\"}"
}
```

#### Metadata Filtering
Filter by custom metadata:
```json
{
  "query": "project tasks",
  "where_filter": {"priority": {"$eq": "high"}},
  "limit": 5
}
```

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start ChromaDB and PostgreSQL
make services

# Run API server locally
uvicorn app.main:app --reload --port 7001
```

### Project Structure
```
api/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── database.py          # Database setup
│   ├── auth.py              # Authentication
│   ├── models.py            # Pydantic models
│   ├── routers/             # API endpoints
│   └── services/            # Business logic
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── Makefile
├── USAGE.md
└── README.md
```

## Production Deployment

### Security Checklist
- [ ] Change default JWT secret key
- [ ] Use strong PostgreSQL passwords
- [ ] Configure CORS origins appropriately
- [ ] Set up SSL/TLS termination
- [ ] Monitor API rate limiting
- [ ] Backup PostgreSQL and ChromaDB data

### Scaling
- Scale API containers horizontally
- Use PostgreSQL connection pooling
- Configure ChromaDB persistence volumes
- Monitor memory usage and performance

## Support

- **Documentation**: http://localhost:7001/docs (Swagger UI)
- **API Reference**: http://localhost:7001/redoc (ReDoc)
- **Health Checks**: http://localhost:7001/api/v1/health