import argparse
import json
import os
from openai import OpenAI
from prompts import EVAL_PROMPT, EVAL_INS

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate QA pairs using OpenAI API.")
    parser.add_argument('--response_dir', type=str, default=r'./results', 
                        help='Path to input file or directory containing JSON files.')
    parser.add_argument('--output_dir', type=str, default=r'./eval_results', 
                        help='Path to save evaluation results.')
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
        {"type": 'text', "text": EVAL_INS}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
        print(f"API Error: {e}")
        return None

def process_file(file_path, output_dir):
    """
    Reads an input file (JSON dict format from inference.py oe mode),
    evaluates all items, and saves the results.
    """
    filename = os.path.basename(file_path)
    print(f"Processing file: {file_path}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    client = OpenAI(api_key=api_key)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    qa_pairs = [(qid, pair) for qid, pair in data.items()]
    print(f"Loaded {len(qa_pairs)} QA pairs from {filename}")
    
    eval_results = []
    output_path = os.path.join(output_dir, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for qid, pair in qa_pairs:
        processed_item = {
            'question_id': qid,
            'question': pair.get('question', ''),
            'answer': pair.get('answer', ''),
            'response': pair.get('model_response', '')
        }
        
        try:
            llm_response = evaluate_single_pair(client, processed_item, EVAL_PROMPT)
            
            if llm_response:
                parsed_result = json.loads(llm_response)
                processed_item['pred'] = parsed_result.get('pred', 'error')
                processed_item['score'] = parsed_result.get('score', -1)
            else:
                raise ValueError("Empty response from LLM")
                
        except Exception as e:
            print(f"Error processing QID {qid}: {e}")
            processed_item['pred'] = 'error'
            processed_item['score'] = -1
        
        eval_results.append(processed_item)
        print(f"[{len(eval_results)}/{len(qa_pairs)}] QID={qid}: pred={processed_item['pred']}, score={processed_item['score']}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=4, ensure_ascii=False)

    valid_results = [r for r in eval_results if r['pred'] != 'error']
    total_valid = len(valid_results)
    
    if total_valid > 0:
        accuracy = sum(r['pred'].lower() == 'yes' for r in valid_results) / total_valid
        avg_score = sum(int(r['score']) for r in valid_results) / total_valid
    else:
        accuracy = 0
        avg_score = 0

    print(f"File: {filename} | Instances: {total_valid} | Accuracy: {accuracy*100:.2f}% | Avg Score: {avg_score:.3f}")

    return eval_results

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

    for file_path in files_to_process:
        try:
            process_file(file_path, args.output_dir)
        except Exception as e:
            print(f"Critical error processing file {file_path}: {e}")


if __name__ == '__main__':
    main()
