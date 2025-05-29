import os
import openai
from openai import AzureOpenAI
import orjson
import argparse
import logging
from process_api_requests_from_file import process_api_requests_from_file
import asyncio

def main(args):
    
    if args.provider == 'openai':
        url = "https://api.openai.com/v1/chat/completions"
        client = openai.OpenAI(api_key=args.api_key)
    elif args.provider == "deepseek":
        url = "https://api.deepseek.com/v1/chat/completions"
    elif args.provider =="openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
    elif args.provider == "google":
        url= "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions" 
    elif args.provider == "azure":
        url = "https://gptreasoners.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview"
    else:
        raise NotImplementedError(f"provider {args.provider} not implemented")
    
    if args.type == "upload":
        if args.input_file is None:
            raise ValueError("input file is required")

        input_file_name = os.path.splitext(os.path.basename(args.input_file))[0]
        previous_directory = os.path.basename(os.path.dirname(args.input_file))
        two_directories_up = os.path.basename(os.path.dirname(os.path.dirname(args.input_file)))
        three_directories_up = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.input_file))))

        path = f"submitted_batches/{three_directories_up}/{two_directories_up}/{previous_directory}"

        if args.batched=="True" and args.provider == "openai":
            batch_file = client.files.create(
                file=open(args.input_file, "rb"),
                purpose="batch"
            )
            batch_job = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            batch_job = client.batches.retrieve(batch_job.id)
            print(f"batch job for file {args.input_file} was created with id {batch_job.id}")
            print(batch_job)
            
            #dump information to a file
            #create a directory if it does not exist
            if not os.path.exists(path):
                os.makedirs(path)

            with open(f"{path}/{input_file_name}.json", "w") as f:
                f.write(orjson.dumps({"id": batch_job.id, "input_file_id_openai":batch_job.input_file_id, "input_file_local": args.input_file, "created_at": batch_job.created_at}).decode('utf-8'))
        
        else:
            path = f"completed_batches/{three_directories_up}/{two_directories_up}/{previous_directory}"
            if not os.path.exists(path):
                os.makedirs(path)
            
            if f"{input_file_name}.jsonl" in os.listdir(path):
                print(f"Warning: {input_file_name}.jsonl already exists in {path}")
                os.remove(os.path.join(path, f"{input_file_name}.jsonl"))
            
            asyncio.run(
                process_api_requests_from_file(requests_filepath=args.input_file,
                                           save_filepath=f"{path}/{input_file_name}.jsonl",
                                           request_url=url,
                                           api_key=args.api_key,
                                           max_requests_per_minute=float(2_000 * 0.5),
                                           max_tokens_per_minute=float(200_000 * 0.5),
                                           token_encoding_name="cl100k_base",
                                           max_attempts=5,
                                           logging_level=int(logging.ERROR))
            )
            
            #sort the responses by id
            with open(f"{path}/{input_file_name}.jsonl", 'r') as f:
                lines = [orjson.loads(line) for line in f]
                lines.sort(key=lambda x: x[0])
                responses = [line for line in lines]

            with open(f"{path}/{input_file_name}.jsonl", 'w') as f:
                for response in responses:
                    res = {"custom_id":response[1]["custom_id"],
                            "response":{"body":response[2]}
                            }
                    f.write(orjson.dumps(res).decode('utf-8') + '\n')
            
    elif args.type == "download":

        data = orjson.loads(open(args.batch_id, 'rb').read())
        batch_id = data["id"]
        input_file = data["input_file_local"]
        input_file_name = os.path.splitext(os.path.basename(input_file))[0]
        previous_directory = os.path.basename(os.path.dirname(input_file))
        two_directories_up = os.path.basename(os.path.dirname(os.path.dirname(input_file)))
        three_directories_up = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(input_file))))

        path = f"completed_batches/{three_directories_up}/{two_directories_up}/{previous_directory}"

        batch_job = client.batches.retrieve(batch_id)
        print(batch_job)
        
        if batch_job.status != "completed":
            print("batch job is not completed yet")
            return
        
        result_file_id = batch_job.output_file_id
        result = client.files.content(result_file_id).content  
        
        if not os.path.exists(path):
            os.makedirs(path)
            
        result_file_name = f"{path}/{input_file_name}.jsonl"
        with open(result_file_name, 'wb') as file:
            file.write(result)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for creating and launching batch jobs for narrative generation using API')
    
    parser.add_argument('--api_key',
                        type=str,
                        default=os.getenv('OPENAI_API_KEY'),
                        help='openai api key')
    parser.add_argument('--input_file',
                        type=str,
                        required=False,
                        help='input file')
    parser.add_argument('--provider',
                        type=str,
                        default='openai',
                        help='provider')
    parser.add_argument('--type',
                        choices=['upload','download'],
                        help='upload or retreive existing batch job')
    parser.add_argument('--batch_id',
                        type=str,
                        required=False,
                        help='batch job id')
    parser.add_argument('--batched',
                        type=str,
                        default="False",
                        help='Whether to use batch inference or not')
    
    args = parser.parse_args()
    main(args)