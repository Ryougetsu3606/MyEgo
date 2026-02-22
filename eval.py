import argparse
import json
import tqdm
import os
import time
import concurrent.futures
from openai import OpenAI
from prompts import eval_prompt, eval_ins

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate QA pairs using OpenAI API.")
    parser.add_argument('--response_dir', type=str, default=r'./result', 
                        help='Path to input file or directory containing JSONL files.')
    parser.add_argument('--output_dir', type=str, default=r'./eval_result', 
                        help='Path to save evaluation results.')
    parser.add_argument('--api_key', type=str, required=True, 
                        help='OpenAI API Key.')
    parser.add_argument('--max_workers', type=int, default=32, 
                        help='Number of concurrent threads for API calls within a file.')
    parser.add_argument('--file_workers', type=int, default=3, 
                        help='Number of concurrent threads for processing multiple files.')
    return parser.parse_args()

def evaluate_single_pair(client, pair, system_prompt):
    """
    Evaluates a single Question-Answer pair using the LLM.
    Returns the parsed JSON result or None if parsing fails.
    """
    user_content = [
        {
            "type": "text",
            "text": (
                f"Please evaluate the following question-answer pair:\n\n" 
                f"Question: {pair['question']}\n" 
                f"Correct Answer: {pair['answer']}\n" 
                f"Predicted Answer: {pair['response']}\n\n" 
            )
        },
        {"type": 'text', "text": eval_ins}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini", # specific model requested
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0,
            stream=False,
        )
        
        result_text = response.choices[0].message.content
        return result_text
        
    except Exception as e:
        print(f"API Error for QID {pair.get('question_id', 'unknown')}: {e}")
        return None

def process_file(file_path, output_dir, api_key, max_workers):
    """
    Reads an input file, evaluates all items, and saves the results.
    """
    start_time = time.time()
    filename = os.path.basename(file_path)
    print(f"Processing file: {file_path}")

    client = OpenAI(api_key=api_key)
    
    requests = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                requests.append(json.loads(line))
    
    print(f"Loaded {len(requests)} QA pairs from {filename}")
    
    eval_results = []
    output_path = os.path.join(output_dir, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(evaluate_single_pair, client, item, eval_prompt): item 
            for item in requests
        }
        
        for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_item), total=len(requests), desc=filename):
            item = future_to_item[future]
            processed_item = item.copy()
            
            try:
                llm_response = future.result()
                
                if llm_response:
                    parsed_result = json.loads(llm_response)
                    processed_item['pred'] = parsed_result.get('pred', 'error')
                    processed_item['score'] = parsed_result.get('score', -1)
                else:
                    raise ValueError("Empty response from LLM")
                    
            except Exception as e:
                print(f"\nError processing item {item.get('question_id')}: {e}")
                processed_item['pred'] = 'error'
                processed_item['score'] = -1
            
            eval_results.append(processed_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=4)

    valid_results = [r for r in eval_results if r['pred'] != 'error']
    total_valid = len(valid_results)
    
    if total_valid > 0:
        accuracy = sum(r['pred'].lower() == 'yes' for r in valid_results) / total_valid
        avg_score = sum(int(r['score']) for r in valid_results) / total_valid
    else:
        accuracy = 0
        avg_score = 0

    duration = time.time() - start_time
    print('\n' + '*' * 100)
    print(f"File: {filename}")
    print(f"Time: {duration:.2f}s | Instances: {total_valid} | Accuracy: {accuracy*100:.2f}% | Avg Score: {avg_score:.3f}")
    print('*' * 100 + '\n')

    return eval_results

def print_summary_table(output_dir):
    """
    Reads all generated output files and prints a summary table.
    """
    print("\nGenerating Summary...")
    print("+" + "-" * 47 + "+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 15 + "+")
    print(f"| {'File':<45} | {'Instances':<13} | {'Accuracy':<10} | {'Avg Score':<13} |")
    print("+" + "-" * 47 + "+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 15 + "+")

    summary_data = []

    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith('.json'):
                file_path = os.path.join(output_dir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    valid_data = [r for r in data if r.get('pred') != 'error']
                    
                    if valid_data:
                        accuracy = sum(r['pred'].lower() == 'yes' for r in valid_data) / len(valid_data)
                        avg_score = sum(int(r['score']) for r in valid_data) / len(valid_data)
                        
                        summary_data.append({
                            'name': file.replace('.json', ''),
                            'count': len(valid_data),
                            'accuracy': accuracy * 100,
                            'score': avg_score
                        })
                except Exception as e:
                    print(f"Could not read summary for {file}: {e}")

    summary_data.sort(key=lambda x: x['accuracy'], reverse=True)

    for item in summary_data:
        print(f"| {item['name']:<45} | {item['count']:<13} | {item['accuracy']:>10.1f} | {item['score']:>12.2f} |")
    
    print("+" + "-" * 47 + "+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 15 + "+")
    print()

def main():
    args = get_args()
    
    files_to_process = []
    if os.path.isfile(args.response_dir):
        files_to_process.append(args.response_dir)
    elif os.path.isdir(args.response_dir):
        files_to_process = [
            os.path.join(args.response_dir, f) 
            for f in os.listdir(args.response_dir) 
            if f.endswith('.json')
        ]
    else:
        print(f"Error: {args.response_dir} is not a valid file or directory.")
        return

    print(f"Found {len(files_to_process)} files to process.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.file_workers) as executor:
        future_to_file = {
            executor.submit(
                process_file, 
                file_path, 
                args.output_dir, 
                args.api_key, 
                args.max_workers
            ): file_path 
            for file_path in files_to_process
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                future.result()
            except Exception as e:
                print(f"Critical error processing file {file_path}: {e}")

    print_summary_table(args.output_dir)

if __name__ == '__main__':
    main()