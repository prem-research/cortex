import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from tqdm import tqdm

from cortex.memory_system import AgenticMemorySystem

load_dotenv()


# Update custom instructions
custom_instructions ="""
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""

# Helper function to extract day from timestamp
def extract_day_from_timestamp(timestamp):
    if not isinstance(timestamp, str) or "on" not in timestamp:
        return None
        
    try:
        parts = timestamp.split("on")
        if len(parts) > 1:
            date_part = parts[1].strip()
            from datetime import datetime
            date_obj = datetime.strptime(date_part, "%d %B, %Y")
            return date_obj.strftime("%A")  # Get day name (Monday, Tuesday, etc.)
    except Exception:
        pass
    return None

class CortexADD:
    def __init__(self, data_path=None, batch_size=2, memory_system=None):
        self.memory_system = memory_system or AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',  # Embedding model
            llm_backend="openai",           # LLM provider
            llm_model="gpt-4o-mini",             # LLM model
            stm_capacity=100                # STM capacity
        )
        
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user_id, message, metadata, retries=3):
        for attempt in range(retries):
            try:
                # Process content to extract metadata if not provided
                analyzed_content = self.memory_system.analyze_content(message)
                
                # Combine provided metadata with analyzed content
                combined_metadata = {
                    "timestamp": metadata.get("timestamp", ""),
                    "context": analyzed_content.get("context", ""),
                    "keywords": analyzed_content.get("keywords", []),
                    "tags": analyzed_content.get("tags", [])
                }

                day = extract_day_from_timestamp(combined_metadata["timestamp"])
                day_info = f" ({day})" if day else ""
                combined_metadata["timestamp"] = combined_metadata["timestamp"] + day_info
                
                # Store only in LTM as specified in requirements
                memory_id = self.memory_system.add_note(
                    content=message,
                    context=combined_metadata["context"],
                    keywords=combined_metadata["keywords"],
                    tags=combined_metadata["tags"],
                    timestamp=combined_metadata["timestamp"],
                    user_id=user_id,
                )
                return memory_id
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)  # Wait before retrying
                    continue
                else:
                    raise e

    def add_memories_batch(self, user_id, messages, timestamp, desc):
        added_memory_ids = []
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc):
            batch_messages = messages[i:i+self.batch_size]
            
            # Create a single batch entry with all messages concatenated
            # This prevents multiple evo_label outputs per batch
            batch_content = "\n\n".join(batch_messages)
            
            # Add as a single memory
            memory_id = self.add_memory(user_id, batch_content, metadata={"timestamp": timestamp})
            added_memory_ids.append(memory_id)
            
        return added_memory_ids

    def process_conversation(self, item, idx):
        conversation = item['conversation']
        speaker_a = conversation['speaker_a']
        speaker_b = conversation['speaker_b']

        # Create a combined user ID for both speakers
        combined_user_id = f"{speaker_a}_{speaker_b}_{idx}"

        # Keep track of all added memory IDs for this conversation
        all_memory_ids = []
        
        for key in conversation.keys():
            if key in ['speaker_a', 'speaker_b'] or "date" in key or "timestamp" in key:
                continue

            date_time_key = key + "_date_time"
            timestamp = conversation.get(date_time_key, "")
            chats = conversation[key]

            # Store all messages from the conversation in a single list
            all_messages = []
            for chat in chats:
                if chat['speaker'] == speaker_a:
                    message = f"{speaker_a}: {chat['text']}"
                elif chat['speaker'] == speaker_b:
                    message = f"{speaker_b}: {chat['text']}"
                else:
                    raise ValueError(f"Unknown speaker: {chat['speaker']}")
                
                all_messages.append(message)

            # Add all messages to memory
            print("SESSION::", combined_user_id)
            memory_ids = self.add_memories_batch(
                combined_user_id, 
                all_messages, 
                timestamp, 
                f"Adding Memories for Conversation {idx} -- {key}"
            )
            all_memory_ids.extend(memory_ids)
            
            # Ensure the results directory exists
            os.makedirs("results", exist_ok=True)
            
            # Save LTM data to JSON, excluding embeddings
            # with open(f"results/memories_{combined_user_id}.json", "w") as f:                
            #     # Get LTM memories for this user - properly include the user_id
            #     memories = self.memory_system.search_memory(
            #         query="",
            #         user_id=combined_user_id,  # Include the user_id!
            #         memory_source="ltm",
            #         limit=1000  # Large number to get all memories
            #     )
                
            #     # Debug: Check if any memories have links
            #     has_links = False
            #     for memory in memories:
            #         links = memory.get("links", {})
            #         if links and links != {} and links != "{}":
            #             has_links = True
            #             # print(f"Found memory with links: {memory['id']}, links: {links}")
                
            #     if not has_links:
            #         print(f"WARNING: No memories found with links for user {combined_user_id}")
                    
            #         # Additional debug: Search other collections
            #         print("Searching other collections for memories with links...")
            #         general_memories = self.memory_system.search_memory(
            #             query="",
            #             user_id=None,  # Try without user filtering
            #             memory_source="ltm",
            #             limit=1000
            #         )
                    
            #         for memory in general_memories:
            #             links = memory.get("links", {})
            #             if links and links != {} and links != "{}":
            #                 print(f"Found memory in general search with links: {memory['id']}, user: {memory.get('user_id')}, links: {links}")
                
            #     # Filter out embeddings and other non-serializable data
            #     serializable_memories = {}
            #     if memories:
            #         for memory in memories:
            #             memory_id = memory.get("id", f"memory_{len(serializable_memories)}")
            #             serializable_memory = {}
            #             for k, v in memory.items():
            #                 if k not in ['embedding', '_embedding']:
            #                     # Handle links specially to ensure they're properly deserialized
            #                     if k == 'links':
            #                         if isinstance(v, str):
            #                             try:
            #                                 v_dict = json.loads(v)
            #                                 serializable_memory[k] = v_dict
            #                                 # Debug specific link serialization
            #                                 print(f"Deserialized links from string to dict: {v} -> {v_dict}")
            #                             except json.JSONDecodeError:
            #                                 serializable_memory[k] = {}
            #                         else:
            #                             serializable_memory[k] = v
            #                     elif isinstance(v, (str, int, float, bool, list, dict)) or v is None:
            #                         serializable_memory[k] = v
            #                     else:
            #                         # Convert non-serializable items to string representation
            #                         serializable_memory[k] = str(v)
            #             serializable_memories[memory_id] = serializable_memory
                
            #     # Also save in a format that we can verify has data
            #     print(f"Saved {len(serializable_memories)} memories to file")
            #     json.dump(serializable_memories, f, indent=4)
        
        # Once all memories are added, trigger a consolidation to ensure proper indexing
        self.memory_system.consolidate_memories()
        
        print("Messages added successfully")

    def process_all_conversations(self, max_workers=10):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.process_conversation, item, idx)
                for idx, item in enumerate(self.data)
            ]

            for future in futures:
                future.result()