import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from tqdm import tqdm

from cortex.memory_system import AgenticMemorySystem
from cortex.constants import (
    DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL, DEFAULT_LLM_BACKEND,
    DEFAULT_STM_CAPACITY, DEFAULT_CHROMA_URI
)
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Main instructions for generating personal memories
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

def extract_day_from_timestamp(timestamp):
    if not isinstance(timestamp, str):
        return None
        
    try:
        if "(" in timestamp and ")" in timestamp:
            day_part = timestamp.split("(")[-1].split(")")[0].strip()
            if day_part:
                return day_part
        
        try:
            from datetime import datetime
            parsed_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return parsed_dt.strftime("%A")
        except (ValueError, AttributeError):
            pass
        
        if "on" in timestamp:
            parts = timestamp.split("on")
            if len(parts) > 1:
                date_part = parts[1].strip()
                if "(" in date_part:
                    date_part = date_part.split("(")[0].strip()
                
                from datetime import datetime
                date_formats = [
                    "%d %B, %Y",
                    "%B %d, %Y",
                    "%d-%m-%Y",
                    "%Y-%m-%d",
                ]
                
                for fmt in date_formats:
                    try:
                        date_obj = datetime.strptime(date_part, fmt)
                        return date_obj.strftime("%A")
                    except ValueError:
                        continue
                        
    except Exception as e:
        logger.warning(f"Could not parse timestamp: {timestamp} - {e}")
        
    return None

class CortexADD:
    def __init__(self, data_path=None, batch_size=2, memory_system=None, enable_background_processing=False):
        if memory_system:
            self.memory_system = memory_system
        else:
            try:
                self.memory_system = AgenticMemorySystem(
                    model_name=DEFAULT_EMBEDDING_MODEL,
                    llm_backend=DEFAULT_LLM_BACKEND,
                    llm_model=DEFAULT_LLM_MODEL,
                    stm_capacity=DEFAULT_STM_CAPACITY,
                    enable_smart_collections=True,
                    enable_background_processing=enable_background_processing
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize AgenticMemorySystem: {e}")
        
        self.enable_background_processing = enable_background_processing
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
        if self.memory_system is None:
            raise RuntimeError("Memory system not initialized")
            
        for attempt in range(retries):
            try:
                timestamp = metadata.get("timestamp", "")
                day = extract_day_from_timestamp(timestamp)
                day_info = f" ({day})" if day else ""
                final_timestamp = timestamp + day_info
                
                memory_id = self.memory_system.add_note(
                    content=message,
                    time=final_timestamp,
                    user_id=user_id,
                )
                return memory_id
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to add memory: {e}")
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"Failed to add memory after {retries} attempts: {e}")
                    raise e

    def add_memories_batch(self, user_id, messages, timestamp, desc):
        added_memory_ids = []
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc):
            batch_messages = messages[i:i+self.batch_size]
            batch_content = "\n\n".join(batch_messages)
            memory_id = self.add_memory(user_id, batch_content, metadata={"timestamp": timestamp})
            added_memory_ids.append(memory_id)
            
        return added_memory_ids

    def process_conversation(self, item, idx):
        conversation = item['conversation']
        speaker_a = conversation['speaker_a']
        speaker_b = conversation['speaker_b']

        combined_user_id = f"{speaker_a}_{speaker_b}_{idx}"
        all_memory_ids = []
        
        for key in conversation.keys():
            if key in ['speaker_a', 'speaker_b'] or "date" in key or "timestamp" in key:
                continue

            date_time_key = key + "_date_time"
            timestamp = conversation.get(date_time_key, "")
            chats = conversation[key]

            all_messages = []
            for chat in chats:
                if chat['speaker'] == speaker_a:
                    message = f"{speaker_a}: {chat['text']}"
                elif chat['speaker'] == speaker_b:
                    message = f"{speaker_b}: {chat['text']}"
                else:
                    raise ValueError(f"Unknown speaker: {chat['speaker']}")
                
                all_messages.append(message)

            print("SESSION::", combined_user_id)
            memory_ids = self.add_memories_batch(
                combined_user_id, 
                all_messages, 
                timestamp, 
                f"Adding Memories for Conversation {idx} -- {key}"
            )
            all_memory_ids.extend(memory_ids)
        
        print("Messages added successfully")

    def process_all_conversations(self, max_workers=4):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        
        if self.memory_system is None:
            raise RuntimeError("Memory system not initialized")
        
        processing_mode = "with background processing" if self.enable_background_processing else "synchronous processing"
        logger.info(f" Processing {len(self.data)} conversations with {max_workers} workers ({processing_mode})")
        
        successful_conversations = 0
        failed_conversations = 0
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for idx, item in enumerate(self.data):
                    future = executor.submit(self._safe_process_conversation, item, idx)
                    futures.append((idx, future))
                
                from concurrent.futures import as_completed
                for idx, future in futures:
                    try:
                        future.result(timeout=720)
                        successful_conversations += 1
                        logger.info(f" Completed conversation {successful_conversations}/{len(self.data)}")
                    except Exception as e:
                        failed_conversations += 1
                        logger.error(f" Error processing conversation {idx}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Parallel processing failed ({e}), falling back to sequential processing")
            successful_conversations = 0
            failed_conversations = 0
            
            for idx, item in enumerate(self.data):
                try:
                    logger.info(f" Sequential processing conversation {idx+1}/{len(self.data)}")
                    self.process_conversation(item, idx)
                    successful_conversations += 1
                except Exception as e:
                    failed_conversations += 1
                    logger.error(f" Error processing conversation {idx}: {e}")
                    continue
                    
        logger.info(f" Processing complete: {successful_conversations} successful, {failed_conversations} failed")
        
        if failed_conversations > 0:
            logger.warning(f" {failed_conversations} conversations failed to process")
            
    def _safe_process_conversation(self, item, idx):
        try:
            time.sleep(0.1 * (idx % 5))
            self.process_conversation(item, idx)
        except Exception as e:
            logger.error(f"Error in _safe_process_conversation for idx {idx}: {e}")
            raise e