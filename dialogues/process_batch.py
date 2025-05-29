import datasets
from tqdm import tqdm
import orjson as json
import os

import argparse

def main(args):
    in_file = args.input_batch
    dial_file = args.dialogue_file

    input_file_name = os.path.splitext(os.path.basename(in_file))[0]
    previous_directory = os.path.basename(os.path.dirname(in_file))
    two_directories_up = os.path.basename(os.path.dirname(os.path.dirname(in_file)))
    three_directories_up = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(in_file))))
    
    ended_count=0
    # open in_file
    with open(in_file, 'r') as f:
        data = f.readlines()

    if dial_file is not None:

        dataset = datasets.load_from_disk(dial_file)
        dialogue = dataset['dialogue']
        models = dataset['models']
        lang = dataset['lang']
        source = dataset['source']
        scene = dataset['scene']
        ended = dataset['ended']

        new = False

    else:
        dialogue =  [[] for _ in range(len(data))]
        models =  [[] for _ in range(len(data))]
        lang =  [[] for _ in range(len(data))]
        source =  [[] for _ in range(len(data))]
        scene = [[] for _ in range(len(data))]
        ended = [[] for _ in range(len(data))]

        new = True
        #read source data
        with open(args.source, 'r') as f:
            source_data = f.readlines()

        assert len(data) == len(source_data)

    dial_file = f"dialogues/{three_directories_up}/{two_directories_up}/{previous_directory}/{input_file_name}"
    
    for i in tqdm(range(len(data))):
        data_input = json.loads(data[i])
        response = data_input["response"]["body"]["choices"][0]["message"]["content"]
        id = int(data_input["custom_id"].split('-')[-1])
        
        if "END_OF_DIALOGUE" in response:
            ended_count+=1
            ended[id] = True
        else:
            dialogue[id].append({"role": args.role, "content": response.strip('"').strip("user: ")}) #strip quotes
            models[id].append(args.model)
        if new:
            src = json.loads(source_data[id])
            lang[id] = args.lang
            source[id] = src["custom_id"]
            scene[id] = src["body"]["messages"][-1]["content"]
            ended[id] = False

    dataset = datasets.Dataset.from_dict({"source": source, "scene": scene, "lang": lang, "dialogue": dialogue, "models": models, "ended": ended})
    dataset.save_to_disk(dial_file)
    print(f"Saved to {dial_file}")
    print(f"INFO: Dialogues that have ended:{100*sum(ended)/len(dataset)}%")
    print(f"INFO: New Dialogues that have ended:{100*ended_count/len(dataset)}%\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for creating and launching batch jobs for narrative generation using VLLM')
    
    parser.add_argument('--dialogue_file',
                        type=str,
                        help='input dialogue dataset. if not provided, will create new one')
    parser.add_argument('--source',
                        type=str,
                        help='source of the dialogue (atomic10x)')
    parser.add_argument('--input_batch',
                        type=str,
                        required=True,
                        help='Input batch completion file, jsonl format')
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help='model used for batch completion')
    parser.add_argument('--role',
                        type=str,
                        required=True,
                        choices=['user','assistant'],
                        help='role of the batch')
    parser.add_argument('--lang',
                        type=str,
                        required=True,
                        help='language of the dialogue')

    args = parser.parse_args()
    if args.dialogue_file is None and args.source is None:
        parser.error("at least one of --dialogue_file and --source required")
    
    main(args)
