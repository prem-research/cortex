import argparse
import os
import time

# Set environment variable to avoid tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.cortex.add import CortexADD
from src.cortex.search import CortexSearch
# from src.langmem import LangMemManager
# from src.memzero.add import MemoryADD
# from src.memzero.search import MemorySearch
from src.openai.predict import OpenAIPredict
from src.rag import RAGManager
from src.utils import METHODS, TECHNIQUES
# from src.zep.add import ZepAdd
# from src.zep.search import ZepSearch
from src.full_context import FullContextPredict


class Experiment:
    def __init__(self, technique_type, chunk_size):
        self.technique_type = technique_type
        self.chunk_size = chunk_size

    def run(self):
        print(f"Running experiment with technique: {self.technique_type}, chunk size: {self.chunk_size}")


def main():
    parser = argparse.ArgumentParser(description='Run memory experiments')
    parser.add_argument('--technique_type', choices=TECHNIQUES, default='mem0',
                        help='Memory technique to use')
    parser.add_argument('--method', choices=METHODS, default='add',
                        help='Method to use')
    parser.add_argument('--search_only', action='store_true', default=False,
                        help='Run search only (skip add phase) - requires existing persistent data')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Chunk size for processing')
    parser.add_argument('--output_folder', type=str, default='results_august_2/',
                        help='Output path for results')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top memories to retrieve')
    parser.add_argument('--filter_memories', action='store_true', default=False,
                        help='Whether to filter memories')
    parser.add_argument('--is_graph', action='store_true', default=False,
                        help='Whether to use graph-based search')
    parser.add_argument('--num_chunks', type=int, default=1,
                        help='Number of chunks to process')

    args = parser.parse_args()

    # Add your experiment logic here
    print(f"Running experiments with technique: {args.technique_type}, chunk size: {args.chunk_size}")

    if args.technique_type == "mem0":
        print("mem0 path not implemented in this workspace.")
        return
    elif args.technique_type == "cortex":
        memory_manager = None
        try:
            # Handle search-only mode (persistent data already exists)
            if args.search_only or args.method == "search":
                print("STARTING CORTEX SEARCH EXPERIMENT (PERSISTENT MODE)")
                output_file_path = os.path.join(
                    args.output_folder,
                    f"cortex_results_top_{args.top_k}_full_dataset.json"
                )
                memory_searcher = CortexSearch(
                    output_file_path,
                    args.top_k,
                    memory_system=None  # Let it create its own connection to persistent vector DB
                )
                memory_searcher.process_data_file_parallel('dataset/locomo10.json', max_workers=10, checkpoint_interval=5)
                print("CORTEX SEARCH EXPERIMENT COMPLETED")
                
            # Handle add phase 
            elif args.method == "add":
                print("STARTING CORTEX ADD EXPERIMENT")
                memory_manager = CortexADD(
                    data_path='dataset/locomo10.json',
                    enable_background_processing=False  # Disable for parallel processing
                )
                memory_manager.process_all_conversations(max_workers=10)
                print("CORTEX ADD EXPERIMENT COMPLETED")
                
        except Exception as e:
            print(f"CORTEX EXPERIMENT ERROR: {e}")
        finally:
            # Ensure proper cleanup
            if memory_manager and hasattr(memory_manager.memory_system, 'shutdown'):
                print("Shutting down memory system...")
                memory_manager.memory_system.shutdown()
    elif args.technique_type == "rag":
        output_file_path = os.path.join(
            args.output_folder,
            f"rag_results_{args.chunk_size}_k{args.num_chunks}.json"
        )
        rag_manager = RAGManager(
            data_path="dataset/locomo10_rag.json",
            chunk_size=args.chunk_size,
            k=args.num_chunks
        )
        rag_manager.process_all_conversations(output_file_path)
    elif args.technique_type == "langmem":
        print("langmem path not implemented in this workspace.")
        return
    elif args.technique_type == "zep":
        print("zep path not implemented in this workspace.")
        return
    elif args.technique_type == "openai":
        output_file_path = os.path.join(args.output_folder, "openai_results.json")
        openai_manager = OpenAIPredict()
        openai_manager.process_data_file("dataset/locomo10.json", output_file_path)
    elif args.technique_type == "fullcontext":
        print("STARTING FULL-CONTEXT EXPERIMENT")
        output_file_path = os.path.join(args.output_folder, "fullcontext_results.json")
        predictor = FullContextPredict()
        predictor.process_data_file_parallel("dataset/locomo10.json", output_file_path, max_workers=10, checkpoint_interval=5)
        print("FULL-CONTEXT EXPERIMENT COMPLETED")
    else:
        raise ValueError(f"Invalid technique type: {args.technique_type}")


if __name__ == "__main__":
    main()
