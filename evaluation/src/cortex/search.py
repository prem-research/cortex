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

load_dotenv()

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
    def __init__(self, output_path='results.json', top_k=10, memory_system=None, temperature_c5=0.5):
        self.memory_system = memory_system or AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',  # Embedding model
            llm_backend="openai",           # LLM provider
            llm_model="gpt-4o-mini",        # LLM model
            stm_capacity=100                # STM capacity
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

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        
        # Analyze the query content to get keywords and context
        analyzed_query = self.memory_system.analyze_content(query)
        # analyzed_query = {"keywords": [], "context": query} #NOTE: NOT USING KEYWORDS FOR NOW
        # print(f"Analyzed query: {analyzed_query}")
        
        while retries < max_retries:
            try:
                # Search only in LTM as specified in requirements
                # Using both content and keywords for search
                content_memories = self.memory_system.search_memory(
                    query=query,
                    user_id=user_id,  # Properly use the user_id
                    memory_source="ltm",  # Search only in LTM
                    limit=4#self.top_k // 2
                )
                # print(f"Content memories: {content_memories[:2]}")
                
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
                print(f"Search error: {e}")
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
                'score': round(memory["score"] if hasattr(memory, 'score') else 0.0, 2),
                'context': memory["context"],
                'keywords': memory["keywords"],
                'tags': memory["tags"]
            }
            formatted_memories.append(memory_dict)
            
            # Get linked memories if available
            if isinstance(memory, dict) and "links" in memory:
                for link_id in memory["links"]:
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
            
        return formatted_memories, linked_memories[:self.top_k], end_time - start_time

    def answer_question(self, speaker_a, speaker_b, idx, question, answer=None, category=None):
        # Create the combined user ID
        combined_user_id = f"{speaker_a}_{speaker_b}_{idx}"
        
        # Perform a single search with the combined user ID
        memories, linked_memories, search_time = self.search_memory(combined_user_id, question)
        self.average_normal_memories.append(len(memories))
        self.average_linked_memories.append(len(linked_memories))
        self.average_memory_time.append(search_time)
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

    def process_questions_parallel(self, qa_list, speaker_a, speaker_b, idx, max_workers=1):
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
        
        print("FINAL AVERAGE MEMORY TIME: ", sum(self.average_memory_time) / len(self.average_memory_time))
        print("FINAL AVERAGE TOKEN COUNT: ", sum(self.num_tokens_list) / len(self.num_tokens_list))

        return results
