# Cortex Memory API Server

A production-ready Python server for the Cortex memory system with FastAPI, OpenAPI documentation, authentication, and gRPC support.

## Features

- 🚀 **FastAPI Server**: High-performance async HTTP API
- 📚 **OpenAPI Documentation**: Interactive API docs at `/docs`
- 🔐 **Authentication**: API key and JWT token authentication
- 🔄 **gRPC Support**: High-performance RPC interface
- 🎯 **Smart Collections**: Automatic memory organization
- ⏱️ **Temporal Search**: Time-aware memory retrieval
- 🔍 **Semantic Search**: AI-powered memory search
- 🏭 **Production Ready**: Docker, rate limiting, monitoring
- 📊 **Metrics**: Prometheus metrics endpoint
- 🔧 **Multi-User Support**: Isolated memory spaces

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cortex-server.git
cd cortex-server
```

2. Copy environment configuration:
```bash
cp server/.env.example server/.env
```

3. Edit `server/.env` and add your OpenAI API key:
```env
OPENAI_API_KEY=your-openai-api-key-here
API_KEYS=your-api-key-1,your-api-key-2
SECRET_KEY=your-secret-key-for-jwt
```

### Running with Docker

1. Start all services:
```bash
docker-compose up -d
```

This starts:
- Cortex API Server (port 8080)
- gRPC Server (port 50051)
- ChromaDB (port 8003)
- Redis (port 6379)
- PostgreSQL (port 5432)
- Nginx proxy (port 80)

2. Check health:
```bash
curl http://localhost:8080/health
```

3. View API documentation:
Open http://localhost:8080/docs in your browser

### Running Locally

1. Install dependencies:
```bash
cd server
pip install -r requirements.txt
```

2. Start ChromaDB:
```bash
docker run -p 8003:8000 chromadb/chroma:latest
```

3. Start the server:
```bash
python run_server.py
```

## API Usage

### Authentication

All API endpoints require authentication. Use one of:

1. **API Key** (Direct):
```bash
curl -H "Authorization: Bearer your-api-key" http://localhost:8080/api/v1/memory
```

2. **JWT Token**:
```bash
# Generate token
curl -X POST http://localhost:8080/auth/token \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your-api-key", "expires_in": 1440}'

# Use token
curl -H "Authorization: Bearer jwt-token" http://localhost:8080/api/v1/memory
```

### Store Memory

```bash
curl -X POST http://localhost:8080/api/v1/memory \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "User prefers TypeScript over JavaScript",
    "context": "programming preferences",
    "tags": ["typescript", "preferences"],
    "user_id": "user_123"
  }'
```

### Search Memories

```bash
curl -X POST http://localhost:8080/api/v1/memory/search \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "programming preferences",
    "limit": 5,
    "memory_source": "all",
    "temporal_weight": 0.3,
    "user_id": "user_123"
  }'
```

### Date Range Search

```bash
curl -X POST http://localhost:8080/api/v1/memory/search \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "recent discussions",
    "date_range": "last week",
    "user_id": "user_123"
  }'
```

## gRPC Usage

### Python Client Example

```python
import grpc
from app.generated import cortex_pb2, cortex_pb2_grpc

# Connect to server
channel = grpc.insecure_channel('localhost:50051')
stub = cortex_pb2_grpc.MemoryServiceStub(channel)

# Store memory
request = cortex_pb2.StoreMemoryRequest(
    content="Test memory",
    context="testing",
    tags=["test"],
    user_id="user_123"
)
response = stub.StoreMemory(request)
print(f"Stored memory with ID: {response.id}")

# Search memories
search_request = cortex_pb2.SearchMemoryRequest(
    query="test",
    limit=5,
    user_id="user_123"
)
search_response = stub.SearchMemories(search_request)
for memory in search_response.memories:
    print(f"Found: {memory.content} (score: {memory.score})")
```

## API Endpoints

### Authentication
- `POST /auth/token` - Generate JWT token from API key

### Memory Operations
- `POST /api/v1/memory` - Store new memory
- `POST /api/v1/memory/search` - Search memories
- `GET /api/v1/memory/{id}` - Get memory by ID
- `PUT /api/v1/memory` - Update memory
- `DELETE /api/v1/memory` - Delete memory
- `POST /api/v1/memory/clear` - Clear memories

### System
- `GET /health` - Health check
- `GET /api/v1/stats` - System statistics
- `GET /metrics` - Prometheus metrics
- `GET /docs` - Interactive API documentation
- `GET /openapi.json` - OpenAPI schema

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Required |
| `API_KEYS` | Comma-separated API keys | Required |
| `SECRET_KEY` | JWT signing secret | Required |
| `HOST` | Server host | 0.0.0.0 |
| `PORT` | Server port | 8080 |
| `WORKERS` | Number of workers | 4 |
| `REDIS_URL` | Redis connection URL | redis://localhost:6379/0 |
| `CHROMA_URI` | ChromaDB URL | http://localhost:8003 |
| `RATE_LIMIT_PER_MINUTE` | API rate limit | 100 |

### Production Deployment

1. **SSL/TLS**: Configure Nginx with SSL certificates
2. **Secrets**: Use environment variables or secret management
3. **Monitoring**: Enable Prometheus metrics and set up Grafana
4. **Scaling**: Adjust worker count based on load
5. **Backup**: Regular backup of ChromaDB and PostgreSQL

## Testing

Run tests:
```bash
cd server
pytest tests/test_api.py -v
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   HTTP Client   │     │   gRPC Client   │
└────────┬────────┘     └────────┬────────┘
         │                       │
    ┌────▼────────────────────────▼────┐
    │         Nginx (Reverse Proxy)    │
    └────┬────────────────────────┬────┘
         │                        │
    ┌────▼──────┐           ┌────▼──────┐
    │  FastAPI  │           │   gRPC    │
    │  (HTTP)   │           │  Server   │
    └────┬──────┘           └────┬──────┘
         │                        │
    ┌────▼────────────────────────▼────┐
    │       Cortex Service Layer       │
    └──────────────┬────────────────────┘
                   │
    ┌──────────────▼────────────────────┐
    │      Cortex Memory System         │
    │  ┌──────────┐  ┌──────────┐      │
    │  │   STM    │  │   LTM    │      │
    │  └──────────┘  └─────┬────┘      │
    │                      │            │
    │  ┌───────────────────▼─────────┐  │
    │  │     ChromaDB (Vectors)      │  │
    │  └─────────────────────────────┘  │
    └───────────────────────────────────┘
              │            │
    ┌─────────▼──┐   ┌────▼──────┐
    │   Redis    │   │ PostgreSQL│
    │  (Cache)   │   │ (API Keys)│
    └────────────┘   └───────────┘
```

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions, please open an issue on GitHub.