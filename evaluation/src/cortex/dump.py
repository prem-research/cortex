#     def find_related_memories_raw(self, query: str, k: int = 5) -> List[MemoryNote]:
#         """Find related memories using hybrid retrieval"""
#         if not self.memories:
#             return []
            
#         # Get indices of related memories
#         # indices = self.retriever.retrieve(query_note.content, k)
#         indices = self.retriever.search(query, k)
        
#         # Convert to list of memories
#         all_memories = list(self.memories.values())
#         memory_str = ""
#         j = 0
#         for i in indices:
#             memory_str +=  "talk start time:" + all_memories[i].timestamp + "memory content: " + all_memories[i].content + "memory context: " + all_memories[i].context + "memory keywords: " + str(all_memories[i].keywords) + "memory tags: " + str(all_memories[i].tags) + "\n"
#             neighborhood = all_memories[i].links
#             for neighbor in neighborhood:
#                 memory_str += "talk start time:" + all_memories[neighbor].timestamp + "memory content: " + all_memories[neighbor].content + "memory context: " + all_memories[neighbor].context + "memory keywords: " + str(all_memories[neighbor].keywords) + "memory tags: " + str(all_memories[neighbor].tags) + "\n"
#                 if j >=k:
#                     break
#                 j += 1
#         return memory_str
    


# # for storage - based on user id store each user's memory separately, with timestamp and analyze_content response for each memory as metadata. Store on LTM only.
# # for retrieval - take the query, do analyze_content on that and use content, and keywords string (separately for both) to do memory search on LTM only. Later for doing memory search there can be a query expansion step.
#                 # Format the retrieved memories properly with talk start time etc, pass it to llm with prompt to answer the actual query.
#                 # later make use of category to do different prompts for different categories.