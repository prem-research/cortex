# Guide on building with Cortex

> special thanks to claude for the initial guide

## What Is This All About?

Think of Cortex as giving your applications a "brain" that can remember things, make connections, and get smarter over time. Just like humans don't forget important conversations or lose context when talking to someone they know, your applications can now have persistent, intelligent memory.

### The Human Memory Analogy

When you meet someone new:
- **Short-term**: You remember their name for the conversation
- **Long-term**: After multiple meetings, you remember their preferences, work style, and history
- **Connections**: You start connecting dots - "Oh, they worked with Sarah, and Sarah mentioned the marketing project..."
- **Context**: When they mention "the Q4 issue," you immediately know what they're talking about

**Cortex does exactly this for your applications.**

---

## What Can You Build?

Here are real examples of what becomes possible:

### 🤖 **Smart Customer Support**
Instead of asking customers to repeat their issue every time:
```
Customer: "Hi, I'm having that database issue again"
Your App: "I see you had a connection timeout with your PostgreSQL setup last week. 
          Is this related to the same server, or a different instance?"
```

### 👨‍💼 **Personal Business Assistant**
```
You: "Schedule a meeting with the marketing team"
Assistant: "I remember you prefer Tuesday mornings, and Sarah from marketing 
           mentioned she's free this week. Shall I suggest Tuesday at 10 AM?"
```

### 🧠 **Knowledge Management System**
```
Employee: "What was our decision about the API rate limiting?"
System: "In March, the team decided to use Redis with a 1000 requests/hour limit.
         This was discussed in the architecture meeting with John and Lisa."
```

### 📚 **Research Assistant**
```
Researcher: "Find papers related to neural networks in healthcare"
Assistant: "I remember you were particularly interested in diagnostic imaging last month.
           Here are 3 recent papers on CNNs for medical diagnosis, plus the 
           transformer-based approach you bookmarked in February."
```

---

## Core Concepts (The Simple Version)

### 1. **Memories vs Database Records**
- **Database**: Static rows and columns
- **Memory**: Living information that connects, evolves, and provides context

### 2. **User Isolation**
Every user has their own completely private memory space. User A's memories never mix with User B's.

### 3. **Sessions**
Think of sessions as "contexts" or "projects":
- `session_id: "customer_support"` - All support-related memories
- `session_id: "project_alpha"` - Everything about project Alpha
- `session_id: ""` - General/default memories

### 4. **Memory Evolution**
Memories automatically:
- **Connect** related information
- **Merge** duplicate or similar content  
- **Strengthen** important relationships over time

---

## Getting Started: Your First Memory-Powered Feature

Let's build a simple "Smart Notes" feature that remembers and connects your thoughts.

### Step 1: Basic Setup

You already have the API running. Here's your first memory interaction:

```bash
# Get your JWT token first
export TOKEN=$(curl -X POST "http://localhost:7001/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}' \
  | jq -r '.access_token')
```

### Step 2: Store Your First Memory

```bash
# Store a work-related thought
curl -X POST "http://localhost:7001/api/v1/memory/add" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The new API endpoint for user authentication is working great. Response time is under 200ms.",
    "session_id": "work_notes",
    "metadata": {
      "topic": "api_development",
      "performance": "good"
    }
  }'
```

### Step 3: Store a Related Memory

```bash
# Store another related thought
curl -X POST "http://localhost:7001/api/v1/memory/add" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Need to add rate limiting to the authentication endpoint to prevent abuse.",
    "session_id": "work_notes",
    "metadata": {
      "topic": "api_development",
      "priority": "high"
    }
  }'
```

### Step 4: Search and See the Magic

```bash
# Search for API-related memories
curl -X POST "http://localhost:7001/api/v1/memory/search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "API authentication performance",
    "session_id": "work_notes",
    "limit": 5
  }'
```

**What happens**: Cortex doesn't just find exact matches. It understands context and returns both memories because they're semantically related - even though one mentions "response time" and the other mentions "rate limiting."

---

## Practical Patterns & Recipes

### Pattern 1: Conversation Memory
Perfect for chatbots, customer support, or personal assistants.

```python
# Python example for a chatbot
import requests
import json

class MemoryBot:
    def __init__(self, base_url, token, user_id):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self.user_id = user_id
    
    def remember_conversation(self, user_message, bot_response, context=None):
        """Store both sides of the conversation"""
        memory_content = f"User said: {user_message}\nBot responded: {bot_response}"
        
        payload = {
            "content": memory_content,
            "session_id": "conversations",
            "metadata": {
                "type": "conversation",
                "context": context or "general"
            }
        }
        
        response = requests.post(f"{self.base_url}/memory/add", 
                               headers=self.headers, json=payload)
        return response.json()
    
    def get_conversation_context(self, current_message):
        """Get relevant context for current conversation"""
        search_payload = {
            "query": current_message,
            "session_id": "conversations",
            "temporal_weight": 0.3,  # Favor recent conversations
            "limit": 3
        }
        
        response = requests.post(f"{self.base_url}/memory/search",
                               headers=self.headers, json=search_payload)
        return response.json()

# Usage
bot = MemoryBot("http://localhost:7001/api/v1", your_token, "user_123")

# In your chat loop
user_input = "I'm having trouble with my account"
context = bot.get_conversation_context(user_input)

# Generate response using context
bot_response = generate_response(user_input, context)

# Remember this interaction
bot.remember_conversation(user_input, bot_response, "support")
```

### Pattern 2: User Preference Learning
Learn and adapt to user preferences over time.

```bash
# Store user preferences as they reveal them
curl -X POST "http://localhost:7001/api/v1/memory/add" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "User prefers morning meetings and short email summaries",
    "session_id": "user_preferences",
    "metadata": {
      "category": "scheduling",
      "confidence": "high"
    }
  }'

# Later, when scheduling:
curl -X POST "http://localhost:7001/api/v1/memory/search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "meeting scheduling preferences",
    "session_id": "user_preferences",
    "limit": 5
  }'
```

### Pattern 3: Knowledge Accumulation
Build up expertise in specific domains.

```bash
# Store domain knowledge
curl -X POST "http://localhost:7001/api/v1/memory/add" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "React hooks like useState should not be called inside loops or conditions. They must be called at the top level.",
    "session_id": "react_knowledge",
    "metadata": {
      "topic": "react",
      "type": "best_practice",
      "difficulty": "intermediate"
    }
  }'

# Query the accumulated knowledge
curl -X POST "http://localhost:7001/api/v1/memory/search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "React hooks best practices",
    "session_id": "react_knowledge",
    "where_filter": {"type": {"$eq": "best_practice"}},
    "limit": 10
  }'
```

### Pattern 4: Temporal Queries
Find information from specific time periods.

```bash
# What happened recently?
curl -X POST "http://localhost:7001/api/v1/memory/search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "project updates",
    "temporal_weight": 0.8,
    "limit": 5
  }'

# What happened in January?
curl -X POST "http://localhost:7001/api/v1/memory/search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "project updates",
    "date_range": "{\"start\": \"2024-01-01T00:00:00Z\", \"end\": \"2024-01-31T23:59:59Z\"}",
    "limit": 5
  }'
```

---

## Best Practices & Pro Tips

### 1. **Session Strategy**
- Use meaningful session names: `customer_support`, `project_alpha`, `user_preferences`
- Keep sessions focused but not too granular
- Empty session (`""`) for general memories

### 2. **Metadata Magic**
Always include relevant metadata - it makes filtering powerful:

```json
{
  "content": "User reported login issue",
  "metadata": {
    "type": "bug_report",
    "severity": "high",
    "component": "authentication",
    "user_id": "12345",
    "status": "investigating"
  }
}
```

### 3. **Search Strategies**

**For Recent Information:**
```json
{"query": "user feedback", "temporal_weight": 0.7}
```

**For Specific Types:**
```json
{"query": "bugs", "where_filter": {"severity": {"$eq": "high"}}}
```

**For Date Ranges:**
```json
{"query": "meetings", "date_range": "last week"}
```

### 4. **Content Quality**
- Store meaningful, complete thoughts rather than fragments
- Include context: "User John from Acme Corp reported..." vs "User reported..."
- Use natural language - Cortex understands context better than keywords

---

## Real Integration Examples

### Example 1: Node.js Express API with Memory

```javascript
const express = require('express');
const axios = require('axios');

class CortexClient {
    constructor(baseUrl, token) {
        this.baseUrl = baseUrl;
        this.headers = { 
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        };
    }

    async remember(content, sessionId = '', metadata = {}) {
        const response = await axios.post(`${this.baseUrl}/memory/add`, {
            content,
            session_id: sessionId,
            metadata
        }, { headers: this.headers });
        
        return response.data;
    }

    async recall(query, sessionId = '', options = {}) {
        const response = await axios.post(`${this.baseUrl}/memory/search`, {
            query,
            session_id: sessionId,
            temporal_weight: options.recentBias || 0.0,
            limit: options.limit || 5,
            where_filter: options.filter
        }, { headers: this.headers });
        
        return response.data.results;
    }
}

const app = express();
const memory = new CortexClient('http://localhost:7001/api/v1', process.env.CORTEX_TOKEN);

app.post('/api/support/ticket', async (req, res) => {
    const { userId, issue, priority } = req.body;
    
    // Store the support ticket
    await memory.remember(
        `User ${userId} reported: ${issue}`,
        'customer_support',
        { 
            type: 'support_ticket',
            priority,
            status: 'open',
            user_id: userId
        }
    );
    
    // Find related historical issues
    const relatedIssues = await memory.recall(
        issue,
        'customer_support',
        { 
            filter: { type: { $eq: 'support_ticket' } },
            limit: 3
        }
    );
    
    res.json({
        ticket_id: generateTicketId(),
        related_issues: relatedIssues.map(r => ({
            content: r.content,
            similarity: r.similarity_score
        }))
    });
});
```

### Example 2: Python Flask Personal Assistant

```python
from flask import Flask, request, jsonify
import requests
from datetime import datetime

class PersonalAssistant:
    def __init__(self, cortex_url, token):
        self.cortex_url = cortex_url
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
    
    def learn_preference(self, preference):
        """Store user preference"""
        return requests.post(f'{self.cortex_url}/memory/add', 
            json={
                'content': preference,
                'session_id': 'preferences',
                'metadata': {
                    'type': 'preference',
                    'learned_at': datetime.now().isoformat()
                }
            }, 
            headers=self.headers
        ).json()
    
    def get_preferences(self, context):
        """Get relevant preferences for a context"""
        return requests.post(f'{self.cortex_url}/memory/search',
            json={
                'query': context,
                'session_id': 'preferences',
                'limit': 5
            },
            headers=self.headers
        ).json()['results']

app = Flask(__name__)
assistant = PersonalAssistant('http://localhost:7001/api/v1', 'your-token')

@app.route('/schedule-meeting', methods=['POST'])
def schedule_meeting():
    meeting_request = request.json['request']
    
    # Get scheduling preferences
    prefs = assistant.get_preferences('meeting scheduling')
    
    # Use preferences to suggest optimal time
    suggestion = generate_meeting_suggestion(meeting_request, prefs)
    
    return jsonify({
        'suggestion': suggestion,
        'based_on_preferences': [p['content'] for p in prefs[:2]]
    })
```

---

## Common Use Cases & Starter Templates

### 1. **Smart FAQ System**
```bash
# Build a FAQ system that learns from interactions
curl -X POST "http://localhost:7001/api/v1/memory/add" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Question: How do I reset my password? Answer: Go to settings > security > reset password. Click the email link to complete reset.",
    "session_id": "faq_knowledge",
    "metadata": {
      "type": "faq",
      "category": "authentication",
      "frequency": 1
    }
  }'
```

### 2. **Learning Chatbot**
```bash
# Store conversation patterns
curl -X POST "http://localhost:7001/api/v1/memory/add" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "When user says they are frustrated, respond with empathy first, then offer specific help",
    "session_id": "conversation_patterns",
    "metadata": {
      "type": "response_pattern",
      "emotion": "frustration"
    }
  }'
```

### 3. **Project Management Assistant**
```bash
# Track project decisions and context
curl -X POST "http://localhost:7001/api/v1/memory/add" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Team decided to use React instead of Vue for the frontend. Main reasons: existing team expertise and component library availability.",
    "session_id": "project_decisions",
    "metadata": {
      "type": "decision",
      "project": "webapp_v2",
      "stakeholders": ["john", "sarah", "mike"]
    }
  }'
```

---

## Troubleshooting & Tips

### Performance Tips
- Use appropriate `limit` values (5-10 for most use cases)
- Use `temporal_weight` when recency matters
- Use `date_range` when you need to filter by time, helps reducing search space and getting more relevant results
- Use metadata filtering to narrow results

### Content Guidelines
- **Good**: "Customer Sarah from Acme Corp reported slow loading times on the dashboard, specifically when viewing reports larger than 1000 rows"
- **Avoid**: "Dashboard slow"

### Session Organization
- **Good**: `customer_support`, `product_feedback`, `team_decisions`
- **Avoid**: `session_1`, `misc`, `data`

### Search Optimization
- Use natural language queries: "authentication problems" vs "auth error"
- Combine search strategies: semantic + temporal + metadata filters
- Start broad, then narrow down with filters

---

## What's Next?

Now that you understand the fundamentals, here are some ideas to explore:

1. **Start Simple**: Build a basic note-taking app that connects related thoughts
2. **Add Intelligence**: Use the search results to power suggestions and context
3. **Scale Up**: Implement user preference learning in your existing app
4. **Get Creative**: Build something unique - maybe a memory-powered game or creative writing assistant

### Questions to Explore
- How could memory enhance your current application?
- What user interactions would benefit from context and history?
- Where do you currently lose context that memory could preserve?

---

## Need Help?

- **Detailed Usage Examples**: See USAGE.md
- **Test your setup**: Use the health endpoint: `GET /api/v1/health`

---

*Happy building! 🚀*