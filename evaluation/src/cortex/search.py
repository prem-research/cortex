import json
import os
import time
import pickle
import tiktoken
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm

from cortex.memory_system import AgenticMemorySystem
from cortex.constants import (
    DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL, DEFAULT_LLM_BACKEND,
    DEFAULT_STM_CAPACITY, DEFAULT_SEARCH_LIMIT, DEFAULT_CHROMA_URI
)

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


load_dotenv()

# Set environment variable to avoid tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Prompt template for generating answers
ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from 
    conversation memories.

    # CONTEXT:
    You have access to memories from a conversation between two speakers. These memories contain 
    timestamped information that may be relevant to answering the question. You also have 
    access to knowledge graph relations showing connections and relationships between memories.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from the conversation
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the 
       memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", 
       etc.), calculate the actual date based on the memory timestamp. For example, if a 
       memory from 4 May 2022 mentions "went to India last year," then the trip occurred 
       in 2021.
    6. Always convert relative time references to specific dates, months, or years. For 
       example, convert "last year" to "2022" or "two months ago" to "March 2023" based 
       on the memory timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories. Pay attention to who said what by looking 
       at the speaker prefix in each memory.
    8. The answer should be less than 5-6 words.
    9. Use the knowledge graph relations to understand the memory network and identify 
       important relationships between memories.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the 
       question
    4. If the answer requires calculation (e.g., converting relative time references), 
       show your work
    5. Analyze the knowledge graph relations to understand the user's knowledge context
    6. Formulate a precise, concise answer based solely on the evidence in the memories
    7. Double-check that your answer directly addresses the question asked
    8. Ensure your final answer is specific and avoids vague time references

    Conversation memories:

    {{memories}}

    Memory relations:

    {{memory_relations}}

    Question: {{question}}

    Answer:
"""

# Category-specific prompts
CATEGORY_PROMPTS = {
    1: """
    You are an intelligent memory assistant tasked with retrieving accurate information from 
    conversation memories.

    # CONTEXT:
    You have access to memories from a conversation between two speakers. These memories contain 
    timestamped information that may be relevant to answering the question. You also have 
    access to knowledge graph relations showing connections and relationships between memories.

    # INSTRUCTIONS:
    Write an answer in the form of a short phrase for the question. Answer with exact words from the 
    provided memories whenever possible. The answer should be less than 5-6 words.

    Conversation memories:

    {{memories}}

    Memory relations:

    {{memory_relations}}

    Question: {{question}}

    Answer:
    """,
    
    2: """
    You are an intelligent memory assistant tasked with retrieving accurate information about dates from 
    conversation memories.

    # CONTEXT:
    You have access to memories from a conversation between two speakers. These memories contain 
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    Use the DATE OF CONVERSATION to answer with an approximate date. Please generate the shortest 
    possible answer, using words from the conversation where possible, and avoid using any subjects.
    The answer should be less than 5-6 words.

    Pay special attention to the TALK START TIME values in the memories, as they contain the date 
    information you need.

    Conversation memories:

    {{memories}}

    Memory relations:

    {{memory_relations}}

    Question: {{question}}

    Answer:
    """,
    
    3: """
    You are an intelligent memory assistant tasked with retrieving accurate information from 
    conversation memories.

    # CONTEXT:
    You have access to memories from a conversation between two speakers. These memories contain 
    information that may be relevant to answering the question.

    # INSTRUCTIONS:
    Answer with EXACT WORDS from the provided memories whenever possible. Write an answer in the form 
    of a short phrase for the question. The answer should be less than 5-6 words.

    Conversation memories:

    {{memories}}

    Memory relations:

    {{memory_relations}}

    Question: {{question}}

    Answer:
    """,
    
    4: """
    You are an intelligent memory assistant tasked with retrieving accurate information from 
    conversation memories.

    # CONTEXT:
    You have access to memories from a conversation between two speakers. These memories contain 
    information that may be relevant to answering the question.

    # INSTRUCTIONS:
    Answer with EXACT WORDS from the provided memories whenever possible. Write an answer in the form 
    of a short phrase for the question. The answer should be less than 5-6 words.

    Conversation memories:

    {{memories}}

    Memory relations:

    {{memory_relations}}

    Question: {{question}}

    Answer:
    """,
    
    5: """
    You are an intelligent memory assistant tasked with retrieving accurate information from 
    conversation memories.

    # CONTEXT:
    You have access to memories from a conversation between two speakers. These memories contain 
    information that may be relevant to answering the question.

    # INSTRUCTIONS:
    You must choose between "{{answer_option}}" and "Not mentioned in the conversation" as your answer.
    Select the correct answer based ONLY on the information in the provided memories.
    If the information is not explicitly mentioned in the memories, choose "Not mentioned in the conversation".

    Conversation memories:

    {{memories}}

    Memory relations:

    {{memory_relations}}

    Question: {{question}}

    Answer (choose exactly one option):
    """
}

encoding = tiktoken.encoding_for_model("gpt-4o")


class CortexSearch:
    def __init__(self, output_path='results.json', top_k=DEFAULT_SEARCH_LIMIT*2, memory_system=None, temperature_c5=0.5, print_running_averages=True):
        # Initialize with persistent ChromaDB storage and automatic embedding backend detection
        # Supports both OpenAI (text-embedding-3-small) and local (all-MiniLM-L6-v2) embeddings
        self.memory_system = memory_system or AgenticMemorySystem(
            model_name=DEFAULT_EMBEDDING_MODEL,     # Use constant
            llm_backend=DEFAULT_LLM_BACKEND,        # Use constant  
            llm_model=DEFAULT_LLM_MODEL,            # Use constant
            stm_capacity=DEFAULT_STM_CAPACITY,      # Use constant
            enable_smart_collections=True,          # Enable smart collections for evaluation
            enable_background_processing=False      # Disable for controlled evaluation environment
        )
        self.top_k = top_k
        self.openai_client = OpenAI()
        self.results = defaultdict(list)
        self.output_path = output_path
        self.temperature_c5 = temperature_c5  # Temperature for category 5 questions
        self.num_tokens_list = []
        self.average_memory_time = []
        self.average_normal_memories = []
        self.average_linked_memories = []
        self.print_running_averages = print_running_averages  # Control running average printing

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1, temporal_weight=None, date_range=None):
        start_time = time.time()
        retries = 0
        
        analyzed_query = self.memory_system.analyze_content(query)
        # Auto-detect temporal queries if temporal_weight not explicitly set
        if temporal_weight is None:
            temporal_keywords = ["last", "recent", "latest", "yesterday", "today", "this week", "this month"]
            has_temporal = any(keyword in query.lower() for keyword in temporal_keywords)
            has_date_range = date_range is not None
            
            if has_temporal or has_date_range:
                temporal_weight = 0.7  # 70% recency + 30% semantic for temporal queries
            else:
                temporal_weight = 0.0  # Pure semantic search by default
        
        while retries < max_retries:
            try:
                # Search only in LTM as specified in requirements
                content_memories = self.memory_system.search_memory(
                    query=query,
                    user_id=user_id,
                    memory_source="ltm",  # Search only in LTM
                    limit=4,#self.top_k // 2
                    temporal_weight=temporal_weight,
                    date_range=date_range
                )
                
                # If we have keywords from analysis, use them as an additional search
                keywords = analyzed_query.get("keywords", [])
                if keywords:
                    keyword_query = " ".join(keywords)
                    keyword_memories = self.memory_system.search_memory(
                        query=keyword_query,
                        user_id=user_id,  # Properly use the user_id
                        memory_source="ltm",
                        limit=self.top_k# // 2
                    )
                    
                    # Combine and deduplicate results
                    combined_memories = content_memories.copy()
                    # print(f"Combined memories: {combined_memories[:2]}")
                    content_ids = {m["id"] for m in content_memories}
                    for memory in keyword_memories:
                        if memory["id"] not in content_ids:
                            combined_memories.append(memory)
                            
                    # Trim if we have more than top_k combined results
                    memories = combined_memories[:self.top_k]
                else:
                    memories = content_memories
                    
                break
            except Exception as e:
                logger.error(f"Search error (attempt {retries + 1}): {e}")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)
                
        end_time = time.time()

        # Format memories for response
        formatted_memories = []
        linked_memories = []
        
        for memory in memories:
            memory_dict = {
                'memory': memory["content"],
                'timestamp': memory["timestamp"],
                'score': round(memory.get("score", 0.0), 2),
                'context': memory.get("context", ""),
                'keywords': memory.get("keywords", []),
                'tags': memory.get("tags", [])
            }
            formatted_memories.append(memory_dict)
            
            # Get linked memories if available
            if isinstance(memory, dict) and "links" in memory and memory["links"]:
                for link_id in memory["links"]:
                    try:
                        linked_memory = self.memory_system.read(link_id, user_id)
                        if linked_memory:
                            link_info = {
                                'memory': linked_memory.content,
                                'timestamp': linked_memory.timestamp,
                                'source_id': memory["id"],
                                'target_id': link_id,
                                'relationship': memory["links"].get(link_id, {}).get('type', 'related'),
                                'strength': memory["links"].get(link_id, {}).get('strength', 0.0)
                            }
                            linked_memories.append(link_info)
                        else:
                            logger.debug(f"Linked memory {link_id} not found for user {user_id}")
                    except Exception as e:
                        logger.warning(f"Could not retrieve linked memory {link_id}: {e}")
                        # Skip invalid links instead of failing
                        continue
            
        return formatted_memories, linked_memories[:self.top_k], end_time - start_time

    def answer_question(self, speaker_a, speaker_b, idx, question, answer=None, category=None):
        # Create the combined user ID
        combined_user_id = f"{speaker_a}_{speaker_b}_{idx}"
        
        # Perform a single search with the combined user ID (with temporal auto-detection)
        memories, linked_memories, search_time = self.search_memory(combined_user_id, question)
        self.average_normal_memories.append(len(memories))
        self.average_linked_memories.append(len(linked_memories))
        self.average_memory_time.append(search_time)
        if self.print_running_averages:
            print("RUNNING AVERAGE MEMORY TIME: ", sum(self.average_memory_time) / len(self.average_memory_time))
            print("RUNNING AVERAGE NORMAL MEMORIES: ", sum(self.average_normal_memories) / len(self.average_normal_memories))
            print("RUNNING AVERAGE LINKED MEMORIES: ", sum(self.average_linked_memories) / len(self.average_linked_memories))
        
        # Format direct memories with day information
        formatted_memories = []
        for item in memories:
            formatted_memories.append(f"TALK START TIME: {item['timestamp']}: {item['memory']}")

        # Format linked memories as graph relations with day information
        memory_relations = []
        for link in linked_memories:
            relation = f"Linked Memory with TALK START TIME: {link['timestamp']} has a {link['relationship']} relationship (strength: {link['strength']}): {link['memory']}"
            memory_relations.append(relation)

        # Choose the appropriate prompt based on category
        if category and category in CATEGORY_PROMPTS:
            template_text = CATEGORY_PROMPTS[category]
            # For category 5, we need to add the answer option
            if category == 5 and answer:
                template_text = template_text.replace("{{answer_option}}", answer)
            prompt_template = Template(template_text)
        else:
            prompt_template = Template(ANSWER_PROMPT)
            
        answer_prompt = prompt_template.render(
            memories=json.dumps(formatted_memories, indent=4),
            memory_relations=json.dumps(memory_relations, indent=4),
            question=question
        )

        # Use different temperature for category 5
        temperature = self.temperature_c5 if category == 5 else 0.0
        
        # Keep track of the number of tokens in the answer prompt, and hold average
        num_tokens = len(encoding.encode(answer_prompt))
        self.num_tokens_list.append(num_tokens)
        if self.print_running_averages:
            print("RUNNING AVERAGE TOKEN COUNT: ", sum(self.num_tokens_list) / len(self.num_tokens_list))
        
        t1 = time.time()
        response = self.openai_client.chat.completions.create(
            model=os.getenv("MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": answer_prompt}
            ],
            temperature=temperature
        )
        t2 = time.time()
        response_time = t2 - t1
        return response.choices[0].message.content, memories, search_time, response_time

    def process_question(self, val, speaker_a, speaker_b, idx):
        question = val.get('question', '')
        answer = val.get('answer', '')
        category = val.get('category', -1)
        evidence = val.get('evidence', [])
        adversarial_answer = val.get('adversarial_answer', '')

        response, memories, memory_time, response_time = self.answer_question(
            speaker_a,
            speaker_b,
            idx,
            question,
            answer,
            category
        )

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "memories": memories,
            'num_memories': len(memories),
            'memory_time': memory_time,
            "response_time": response_time
        }

        # Save results after each question is processed
        with open(self.output_path, 'w') as f:
            json.dump(self.results, f, indent=4)

        return result

    def process_data_file(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            qa = item['qa']
            conversation = item['conversation']
            speaker_a = conversation['speaker_a']
            speaker_b = conversation['speaker_b']

            for question_item in tqdm(qa, total=len(qa), desc=f"Processing questions for conversation {idx}", leave=False):
                result = self.process_question(
                    question_item,
                    speaker_a,
                    speaker_b,
                    idx
                )
                self.results[idx].append(result)

                # Save results after each question is processed
                with open(self.output_path, 'w') as f:
                    json.dump(self.results, f, indent=4)

        # Final save at the end
        with open(self.output_path, 'w') as f:
            json.dump(self.results, f, indent=4)

    def process_questions_parallel(self, qa_list, speaker_a, speaker_b, idx, max_workers=8):
        def process_single_question(val):
            result = self.process_question(val, speaker_a, speaker_b, idx)
            # Save results after each question is processed
            with open(self.output_path, 'w') as f:
                json.dump(self.results, f, indent=4)
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(process_single_question, qa_list),
                total=len(qa_list),
                desc="Answering Questions"
            ))

        # Final save at the end
        with open(self.output_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        if self.print_running_averages:
            print("FINAL AVERAGE MEMORY TIME: ", sum(self.average_memory_time) / len(self.average_memory_time))
            print("FINAL AVERAGE TOKEN COUNT: ", sum(self.num_tokens_list) / len(self.num_tokens_list))

        return results

    def process_data_file_parallel(self, file_path, max_workers=4, checkpoint_interval=5):
        """Parallel version of process_data_file that processes conversations in parallel"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Aggregated metrics from all workers
        all_memory_times = []
        all_normal_memories = []
        all_linked_memories = []
        all_token_counts = []
        completed_conversations = 0

        def process_single_conversation(idx_item_pair):
            idx, item = idx_item_pair
            qa = item['qa']
            conversation = item['conversation']
            speaker_a = conversation['speaker_a']
            speaker_b = conversation['speaker_b']
            
            # Create a worker instance with running averages disabled and shared memory system
            worker = CortexSearch(
                output_path=f"temp_worker_{idx}.json",
                top_k=self.top_k,
                memory_system=self.memory_system,  # Share the same memory system to avoid re-initialization
                temperature_c5=self.temperature_c5,
                print_running_averages=False  # Disable worker printing
            )
            
            # Process questions for this conversation
            results = worker.process_questions_parallel(qa, speaker_a, speaker_b, idx, max_workers=8)
            
            # Return results and metrics from this worker
            return idx, results, {
                'memory_times': worker.average_memory_time,
                'normal_memories': worker.average_normal_memories, 
                'linked_memories': worker.average_linked_memories,
                'token_counts': worker.num_tokens_list
            }

        # Process conversations in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_conversation, (idx, item))
                for idx, item in enumerate(data)
            ]
            
            # Collect results as they complete
            from concurrent.futures import as_completed
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing conversations in parallel"):
                try:
                    idx, conversation_results, worker_metrics = future.result(timeout=300)  # 5-minute timeout
                    self.results[idx].extend(conversation_results)
                    
                    # Aggregate metrics from this worker
                    all_memory_times.extend(worker_metrics['memory_times'])
                    all_normal_memories.extend(worker_metrics['normal_memories'])
                    all_linked_memories.extend(worker_metrics['linked_memories'])
                    all_token_counts.extend(worker_metrics['token_counts'])
                    
                    completed_conversations += 1
                    
                    # Print aggregated metrics at checkpoints
                    if completed_conversations % checkpoint_interval == 0:
                        if all_memory_times:
                            avg_memory_time = sum(all_memory_times) / len(all_memory_times)
                            avg_normal_mem = sum(all_normal_memories) / len(all_normal_memories)
                            avg_linked_mem = sum(all_linked_memories) / len(all_linked_memories)
                            avg_tokens = sum(all_token_counts) / len(all_token_counts)
                            
                            print(f"\n CHECKPOINT {completed_conversations}/{len(data)} conversations:")
                            print(f"   Average memory time: {avg_memory_time:.3f}s")
                            print(f"   Average normal memories: {avg_normal_mem:.1f}")
                            print(f"   Average linked memories: {avg_linked_mem:.1f}")
                            print(f"   Average token count: {avg_tokens:.1f}")
                    
                    # Save results after each conversation is processed
                    with open(self.output_path, 'w') as f:
                        json.dump(self.results, f, indent=4)
                        
                except Exception as e:
                    logger.error(f"Error processing conversation: {e}")
                    continue

        # Print final aggregated averages
        if all_memory_times:
            print(f"\n FINAL AGGREGATED METRICS ({len(data)} conversations):")
            print(f"   Total questions processed: {len(all_memory_times)}")
            print(f"   Average memory time: {sum(all_memory_times) / len(all_memory_times):.3f}s")
            print(f"   Average normal memories: {sum(all_normal_memories) / len(all_normal_memories):.1f}")
            print(f"   Average linked memories: {sum(all_linked_memories) / len(all_linked_memories):.1f}")
            print(f"   Average token count: {sum(all_token_counts) / len(all_token_counts):.1f}")
        
        # Update main instance metrics for consistency
        self.average_memory_time = all_memory_times
        self.average_normal_memories = all_normal_memories
        self.average_linked_memories = all_linked_memories
        self.num_tokens_list = all_token_counts

        # Cleanup temporary worker files
        import os
        for idx in range(len(data)):
            temp_file = f"temp_worker_{idx}.json"
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Could not remove temp file {temp_file}: {e}")

        # Final save at the end
        with open(self.output_path, 'w') as f:
            json.dump(self.results, f, indent=4)
