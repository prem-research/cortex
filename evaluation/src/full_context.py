import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

FULL_CONTEXT_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from 
conversation memories.

# CONTEXT:
You have access to memories from a conversation between two speakers. These memories contain 
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from the conversation
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.),
   calculate the actual date based on the memory timestamp. For example, if a memory from
   4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example,
   convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
   timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories. Pay attention to who said what by looking
   at the speaker prefix in each memory.
8. The answer should be less than 5-6 words.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Conversation memories:

{{memories}}

Question: {{question}}

Answer:
"""


class FullContextPredict:
    """Full-context evaluator that bypasses retrieval and feeds the entire conversation
    (all turns with timestamps) directly to the LLM for each question.
    """

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("MODEL", "gpt-4o")
        self.openai_client = OpenAI()
        self.results = defaultdict(list)

    @staticmethod
    def build_full_context(conversation: dict) -> str:
        """Build a single textual context from the raw conversation structure.
        Expects keys like 'speaker_a', 'speaker_b', per-turn message arrays, and
        corresponding '<turn>_date_time' entries.
        """
        lines: list[str] = []
        for key, value in conversation.items():
            if key in ("speaker_a", "speaker_b"):
                continue
            # Skip timestamp keys themselves; we will fetch them alongside the turn
            if key.endswith("_date_time"):
                continue

            timestamp = conversation.get(f"{key}_date_time", "")
            chats = value if isinstance(value, list) else []
            for chat in chats:
                speaker = chat.get("speaker", "")
                text = chat.get("text", "")
                if text:
                    lines.append(f"TALK START TIME: {timestamp}: {speaker}: {text}")
        return "\n".join(lines)

    def answer_question(self, conversation: dict, question: str):
        context = self.build_full_context(conversation)
        template = Template(FULL_CONTEXT_PROMPT)
        answer_prompt = template.render(memories=context, question=question)

        t1 = time.time()
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": answer_prompt}],
            temperature=0.0,
        )
        t2 = time.time()
        response_time = t2 - t1
        return response.choices[0].message.content, response_time, context

    def process_data_file(self, file_path: str, output_file_path: str, max_workers: int = 0):
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Sequential by default to keep rate limits predictable
        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations (full-context)"):
            qa = item.get('qa', [])
            conversation = item.get('conversation', {})

            for question_item in tqdm(qa, total=len(qa), desc=f"Questions for conversation {idx}", leave=False):
                question = question_item.get('question', '')
                answer = question_item.get('answer', '')
                category = question_item.get('category', -1)
                evidence = question_item.get('evidence', [])
                adversarial_answer = question_item.get('adversarial_answer', '')

                response, response_time, context = self.answer_question(conversation, question)

                result = {
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "evidence": evidence,
                    "response": response,
                    "adversarial_answer": adversarial_answer,
                    "response_time": response_time,
                    "context": context,
                }
                self.results[idx].append(result)

                with open(output_file_path, 'w') as f:
                    json.dump(self.results, f, indent=4)

        # Final save
        with open(output_file_path, 'w') as f:
            json.dump(self.results, f, indent=4)

    def process_data_file_parallel(self, file_path: str, output_file_path: str, max_workers: int = 10, checkpoint_interval: int = 5):
        """Process conversations in parallel, one thread per conversation.
        Each worker iterates its conversation's questions sequentially to keep
        per-conversation context stable while achieving higher throughput overall.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        def process_single(idx_item_pair):
            idx, item = idx_item_pair
            qa = item.get('qa', [])
            conversation = item.get('conversation', {})
            conv_results = []
            for question_item in qa:
                question = question_item.get('question', '')
                answer = question_item.get('answer', '')
                category = question_item.get('category', -1)
                evidence = question_item.get('evidence', [])
                adversarial_answer = question_item.get('adversarial_answer', '')

                response, response_time, context = self.answer_question(conversation, question)
                conv_results.append({
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "evidence": evidence,
                    "response": response,
                    "adversarial_answer": adversarial_answer,
                    "response_time": response_time,
                    "context": context,
                })
            return idx, conv_results

        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single, (idx, item)) for idx, item in enumerate(data)]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Conversations (parallel)"):
                idx, conv_results = future.result()
                self.results[idx].extend(conv_results)
                completed += 1

                if completed % checkpoint_interval == 0:
                    with open(output_file_path, 'w') as f:
                        json.dump(self.results, f, indent=4)

        # Final save
        with open(output_file_path, 'w') as f:
            json.dump(self.results, f, indent=4)
