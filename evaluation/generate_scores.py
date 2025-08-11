import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import argparse
from pathlib import Path
import numpy as np

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

# Define the results directory
results_dir = Path('results_august')

# Performance metrics provided by user
performance_metrics = {
    10: {'memory_time': 8.32, 'questions': 1986, 'normal_memories': 10.0, 'linked_memories': 9.9, 'token_count': 1982.3},
    15: {'memory_time': 8.55, 'questions': 1986, 'normal_memories': 15.0, 'linked_memories': 14.9, 'token_count': 2900.8},
    20: {'memory_time': 8.88, 'questions': 1986, 'normal_memories': 20.0, 'linked_memories': 19.9, 'token_count': 3824.3},
    25: {'memory_time': 8.73, 'questions': 1986, 'normal_memories': 25.0, 'linked_memories': 25.0, 'token_count': 4745.6},
    30: {'memory_time': 9.04, 'questions': 1986, 'normal_memories': 30.0, 'linked_memories': 30.0, 'token_count': 5662.5},
    35: {'memory_time': 9.18, 'questions': 1986, 'normal_memories': 35.0, 'linked_memories': 35.0, 'token_count': 6569.0},
    40: {'memory_time': 9.31, 'questions': 1986, 'normal_memories': 40.0, 'linked_memories': 40.0, 'token_count': 7480.1},
    45: {'memory_time': 9.47, 'questions': 1986, 'normal_memories': 45.0, 'linked_memories': 45.0, 'token_count': 8397.5},
    50: {'memory_time': 9.62, 'questions': 1986, 'normal_memories': 50.0, 'linked_memories': 50.0, 'token_count': 9304.2},
    55: {'memory_time': 9.02, 'questions': 1986, 'normal_memories': 55.0, 'linked_memories': 55.0, 'token_count': 10220.3},
}

def extract_top_k_from_filename(filename):
    """Extract top_k value from filename like 'result_cortex_results_top_20_full_dataset.json'"""
    match = re.search(r'top_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def process_result_file(filepath):
    """Process a single result file and return overall scores"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Flatten the data into a list of question items
        all_items = []
        for key in data:
            all_items.extend(data[key])
        
        # Convert to DataFrame
        df = pd.DataFrame(all_items)
        
        # Convert category to numeric type
        df['category'] = pd.to_numeric(df['category'])
        
        # Calculate overall means
        overall_means = df.agg({
            'bleu_score': 'mean',
            'f1_score': 'mean',
            'llm_score': 'mean'
        }).round(4)
        
        # Calculate category-wise means
        category_means = df.groupby('category').agg({
            'bleu_score': 'mean',
            'f1_score': 'mean',
            'llm_score': 'mean'
        }).round(4)
        
        return {
            'overall': overall_means.to_dict(),
            'by_category': category_means.to_dict(),
            'total_questions': len(df)
        }
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def run_original_analysis(file_path):
    """Run the original single-file analysis (backward compatibility)"""
    print(f"Running original analysis on: {file_path}")
    
    # Load the evaluation metrics data
    with open(file_path, 'r') as f:
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

def run_comprehensive_analysis():
    """Run the comprehensive analysis on all result files"""
    # Find all result files
    result_files = []
    for file in results_dir.glob('result_cortex_results_top_*_full_dataset.json'):
        top_k = extract_top_k_from_filename(file.name)
        if top_k is not None:
            result_files.append((top_k, file))
    
    # Sort by top_k value
    result_files.sort(key=lambda x: x[0])
    
    print(f"Found {len(result_files)} result files")
    
    # Process all files and collect data
    results_data = []
    
    for top_k, filepath in result_files:
        print(f"Processing top_k={top_k}: {filepath.name}")
        
        result = process_result_file(filepath)
        if result:
            # Combine with performance metrics
            perf_metrics = performance_metrics.get(top_k, {})
            
            data_point = {
                'top_k': top_k,
                'bleu_score': result['overall']['bleu_score'],
                'f1_score': result['overall']['f1_score'],
                'llm_score': result['overall']['llm_score'],
                'total_questions': result['total_questions'],
                'memory_time': perf_metrics.get('memory_time', None),
                'normal_memories': perf_metrics.get('normal_memories', None),
                'linked_memories': perf_metrics.get('linked_memories', None),
                'token_count': perf_metrics.get('token_count', None)
            }
            results_data.append(data_point)
            
            print(f"  Overall scores - BLEU: {data_point['bleu_score']:.4f}, "
                  f"F1: {data_point['f1_score']:.4f}, LLM: {data_point['llm_score']:.4f}")
    
    # Convert to DataFrame for easier plotting
    df_results = pd.DataFrame(results_data)
    
    # Print summary table
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)
    print(df_results.to_string(index=False, float_format='%.4f'))
    
    # Create comprehensive visualization with 4 graphs
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Overall Scores vs Top K
    plt.subplot(2, 2, 1)
    plt.plot(df_results['top_k'], df_results['llm_score'], 'o-', label='LLM Score', linewidth=2, markersize=8)
    plt.plot(df_results['top_k'], df_results['bleu_score'], 's-', label='BLEU Score', linewidth=2, markersize=8)
    plt.plot(df_results['top_k'], df_results['f1_score'], '^-', label='F1 Score', linewidth=2, markersize=8)
    plt.xlabel('Top K')
    plt.ylabel('Score')
    plt.title('Overall Scores vs Top K', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. LLM Score vs Top K (detailed)
    plt.subplot(2, 2, 2)
    plt.plot(df_results['top_k'], df_results['llm_score'], 'o-', color='red', linewidth=3, markersize=10)
    plt.xlabel('Top K')
    plt.ylabel('LLM Score')
    plt.title('LLM Score vs Top K', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    for i, row in df_results.iterrows():
        plt.annotate(f'{row["llm_score"]:.3f}', 
                    (row['top_k'], row['llm_score']), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # 3. Score Efficiency (LLM Score / Memory Time)
    plt.subplot(2, 2, 3)
    valid_efficiency_data = df_results.dropna(subset=['memory_time'])
    efficiency = valid_efficiency_data['llm_score'] / valid_efficiency_data['memory_time']
    plt.plot(valid_efficiency_data['top_k'], efficiency, 'v-', color='brown', linewidth=3, markersize=10)
    plt.xlabel('Top K')
    plt.ylabel('LLM Score / Memory Time')
    plt.title('Score Efficiency vs Top K', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    for i, row in valid_efficiency_data.iterrows():
        eff_val = row['llm_score'] / row['memory_time']
        plt.annotate(f'{eff_val:.3f}', 
                    (row['top_k'], eff_val), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # 4. Token Count vs LLM Score
    plt.subplot(2, 2, 4)
    valid_token_data = df_results.dropna(subset=['token_count'])
    plt.scatter(valid_token_data['token_count'], valid_token_data['llm_score'], 
               c=valid_token_data['top_k'], cmap='viridis', s=100, alpha=0.8)
    plt.colorbar(label='Top K')
    plt.xlabel('Average Token Count')
    plt.ylabel('LLM Score')
    plt.title('Token Count vs LLM Score', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    # Add annotations for each point
    for i, row in valid_token_data.iterrows():
        plt.annotate(f'K={int(row["top_k"])}', 
                    (row['token_count'], row['llm_score']), 
                    textcoords="offset points", xytext=(5,5), ha='left', fontsize=10)
    
    plt.tight_layout()
    comprehensive_path = results_dir / 'cortex_evaluation_comprehensive_analysis.png'
    plt.savefig(comprehensive_path, dpi=300, bbox_inches='tight')
    print(f"\nComprehensive analysis saved as '{comprehensive_path}'")
    
    # Create individual focused plots
    # 1. Overall Scores vs Top K
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['top_k'], df_results['llm_score'], 'o-', label='LLM Score', linewidth=3, markersize=10)
    plt.plot(df_results['top_k'], df_results['bleu_score'], 's-', label='BLEU Score', linewidth=3, markersize=10)
    plt.plot(df_results['top_k'], df_results['f1_score'], '^-', label='F1 Score', linewidth=3, markersize=10)
    plt.xlabel('Top K', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Overall Scores vs Top K', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    overall_path = results_dir / 'overall_scores_vs_top_k.png'
    plt.savefig(overall_path, dpi=300, bbox_inches='tight')
    
    # 2. LLM Score vs Top K
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['top_k'], df_results['llm_score'], 'o-', color='red', linewidth=4, markersize=12)
    plt.xlabel('Top K', fontsize=14)
    plt.ylabel('LLM Score', fontsize=14)
    plt.title('LLM Score vs Top K', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    for i, row in df_results.iterrows():
        plt.annotate(f'{row["llm_score"]:.4f}', 
                    (row['top_k'], row['llm_score']), 
                    textcoords="offset points", xytext=(0,15), ha='center', fontsize=12)
    plt.tight_layout()
    llm_path = results_dir / 'llm_score_vs_top_k.png'
    plt.savefig(llm_path, dpi=300, bbox_inches='tight')
    
    # 3. Score Efficiency
    plt.figure(figsize=(10, 6))
    valid_efficiency_data = df_results.dropna(subset=['memory_time'])
    efficiency = valid_efficiency_data['llm_score'] / valid_efficiency_data['memory_time']
    plt.plot(valid_efficiency_data['top_k'], efficiency, 'v-', color='brown', linewidth=4, markersize=12)
    plt.xlabel('Top K', fontsize=14)
    plt.ylabel('Score Efficiency (LLM Score / Memory Time)', fontsize=14)
    plt.title('Score Efficiency vs Top K', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    for i, row in valid_efficiency_data.iterrows():
        eff_val = row['llm_score'] / row['memory_time']
        plt.annotate(f'{eff_val:.4f}', 
                    (row['top_k'], eff_val), 
                    textcoords="offset points", xytext=(0,15), ha='center', fontsize=12)
    plt.tight_layout()
    efficiency_path = results_dir / 'score_efficiency_vs_top_k.png'
    plt.savefig(efficiency_path, dpi=300, bbox_inches='tight')
    
    # 4. Token Count vs LLM Score
    plt.figure(figsize=(10, 6))
    valid_token_data = df_results.dropna(subset=['token_count'])
    scatter = plt.scatter(valid_token_data['token_count'], valid_token_data['llm_score'], 
                         c=valid_token_data['top_k'], cmap='viridis', s=150, alpha=0.8, edgecolors='black')
    plt.colorbar(scatter, label='Top K')
    plt.xlabel('Average Token Count', fontsize=14)
    plt.ylabel('LLM Score', fontsize=14)
    plt.title('Token Count vs LLM Score', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    # Add annotations for each point
    for i, row in valid_token_data.iterrows():
        plt.annotate(f'K={int(row["top_k"])}', 
                    (row['token_count'], row['llm_score']), 
                    textcoords="offset points", xytext=(8,8), ha='left', fontsize=11, fontweight='bold')
    plt.tight_layout()
    token_llm_path = results_dir / 'token_count_vs_llm_score.png'
    plt.savefig(token_llm_path, dpi=300, bbox_inches='tight')
    
    print(f"\nIndividual plots saved in {results_dir}/:")
    print(f"- {overall_path}")
    print(f"- {llm_path}")
    print(f"- {efficiency_path}")
    print(f"- {token_llm_path}")
    
    # Save the processed data as CSV for further analysis
    csv_path = results_dir / 'cortex_evaluation_results.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"- {csv_path} (data file)")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Generate evaluation scores and visualizations')
    parser.add_argument('--mode', choices=['original', 'comprehensive'], default='comprehensive',
                       help='Mode: "original" for single file analysis, "comprehensive" for all files analysis')
    parser.add_argument('--file', type=str, 
                       default='results_july/result_cortex_results_top_40_full_dataset.json',
                       help='File path for original mode analysis')
    
    args = parser.parse_args()
    
    if args.mode == 'original':
        run_original_analysis(args.file)
    else:
        run_comprehensive_analysis()

if __name__ == "__main__":
    main()