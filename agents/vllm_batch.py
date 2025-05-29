import os
from vllm import LLM, SamplingParams
import orjson
import argparse


def main(args):
    if args.input_file is None:
        raise ValueError("input file is required")
    if args.output is None:
        input_file_name = os.path.splitext(os.path.basename(args.input_file))[0]
        previous_directory = os.path.basename(os.path.dirname(args.input_file))
        two_directories_up = os.path.basename(os.path.dirname(os.path.dirname(args.input_file)))
        three_directories_up = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.input_file))))
        out_fp = f"completed_batches/{three_directories_up}/{two_directories_up}/{previous_directory}/{input_file_name}.jsonl"
        if not os.path.exists(os.path.dirname(out_fp)):
            os.makedirs(os.path.dirname(out_fp))
    else:
        out_fp = args.output
    
    custom_ids = []
    prompts_to_process = []
    with open(out_fp, 'w') as file_out:
        with open(args.input_file, 'r') as file:
            first_line=True
            for line in file:
                job = orjson.loads(line)
                custom_ids.append(job["custom_id"])
                if first_line:
                    # Initialize model and tokenizer from job
                    first_line = False
                    model = job["body"]["model"]
                    llm = LLM(model=model,
                                tensor_parallel_size=args.tensor_parallel_size,
                                pipeline_parallel_size=args.pipeline_parallel_size,
                                max_model_len=2048*2,
                                trust_remote_code=True,
                                #enable_prefix_caching=True, # Phi doesn't support APC
                                gpu_memory_utilization=args.util)
                    tokenizer = llm.get_tokenizer()
                    sampling_params = SamplingParams(temperature=job["body"]["temperature"],
                                            max_tokens=job["body"]["max_tokens"],
                                            top_p=job["body"]["top_p"],
                                            #frequency_penalty=job["body"]["frequency_penalty"],
                                            #presence_penalty=job["body"]["presence_penalty"]
                                            )
                
                if "gemma-3" in model:
                    message = add_type(job["body"]["messages"])
                else:
                    message = job["body"]["messages"]
                prompts_to_process.append(tokenizer.apply_chat_template(message,
                                        tokenize=False,
                                        add_generation_prompt=True))
        
        outputs = llm.generate(prompts_to_process, sampling_params)
        
        for i,output in enumerate(outputs):
            obj = {
                "custom_id": custom_ids[i],
                "response": {
                    "body": {
                        "model": job["body"]["model"],
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": output.outputs[0].text
                                },
                            },
                        ],
                    },
                "prompt": output.prompt,
                }
            }
            file_out.write(orjson.dumps(obj).decode('utf-8','replace') + '\n')
    print(f"Saved to {out_fp}")


def add_type(messages):
    """
    Transforms the list of message dictionaries by wrapping each message's content string
    in a list containing a dictionary with explicit type information.

    Args:
        messages (list): A list of dictionaries, where each dictionary has keys "role" and "content".

    Returns:
        list: A new list of messages where the "content" value is a list of dictionaries with keys
              "type" and "text".
    """
    return [
        {
            "role": msg["role"],
            "content": [{"type": "text", "text": msg["content"]}],
        }
        for msg in messages
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for creating and launching batch jobs for narrative generation using VLLM')
    
    parser.add_argument('--input_file',
                        type=str,
                        required=False,
                        help='input file')
    parser.add_argument('--output',
                        type=str,
                        help='Path to output file')
    parser.add_argument('--tensor_parallel_size',
                        type=int,
                        default=4)
    parser.add_argument('--pipeline_parallel_size',
                        type=int,
                        default=1)
    parser.add_argument('--util',
                        type=float,
                        default=0.9)
    
    args = parser.parse_args()
    main(args)
    
