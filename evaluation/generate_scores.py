import json

import pandas as pd

# Load the evaluation metrics data
# with open('results_without_cat/result_cortex_results_top_10_full_dataset.json', 'r') as f:
# with open('results_v1/result_cortex_results_top_20_full_dataset.json', 'r') as f:
# with open('results/result_cortex_results_top_20_full_dataset.json', 'r') as f:
# with open('results/result_cortex_results_top_15_full_dataset.json', 'r') as f:
with open('results/result_cortex_results_top_30_full_dataset.json', 'r') as f:
    data = json.load(f)

# Flatten the data into a list of question items
all_items = []
for key in data:
    all_items.extend(data[key])

# Convert to DataFrame
df = pd.DataFrame(all_items)

# Convert category to numeric type
df['category'] = pd.to_numeric(df['category'])

# Calculate mean scores by category
result = df.groupby('category').agg({
    'bleu_score': 'mean',
    'f1_score': 'mean',
    'llm_score': 'mean'
}).round(4)

# Add count of questions per category
result['count'] = df.groupby('category').size()

# Print the results
print("Mean Scores Per Category:")
print(result)

# Calculate overall means
overall_means = df.agg({
    'bleu_score': 'mean',
    'f1_score': 'mean',
    'llm_score': 'mean'
}).round(4)

print("\nOverall Mean Scores:")
print(overall_means)