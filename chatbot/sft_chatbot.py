from trl import SFTTrainer,SFTConfig
from peft import LoraConfig
from transformers import HfArgumentParser
from datasets import load_dataset

from argparse import ArgumentParser


def arguments():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Ministral-8B-Instruct-2410")
    parser.add_argument("--dataset", type=str, default="Johndfm/medal_chatbot_responses")
    parser.add_argument(
        "--args_file",
        type=str,
        default="sft_config.yaml",
        help="Path to the YAML file containing SFT training arguments.",
    )
    return parser.parse_args()



def main(args):
    
    sft_args,  = HfArgumentParser(SFTConfig).parse_yaml_file(args.args_file)
    
    def preprocess_function(example):
            return {
                "prompt": example["context"],
                "completion": [example["response"]],
            }

    dataset = load_dataset(args.dataset)
    dataset = dataset.filter(lambda x: x["role"] == "assistant")
    dataset = dataset.map(preprocess_function, remove_columns=["context","response","source", "scene", "lang", "model","role"])

    print(next(iter(dataset["train"])))
    
    trainer = SFTTrainer(
        model=args.model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=LoraConfig(
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
        ),
        args=sft_args
    )
    trainer.train()


if __name__ == "__main__":
    args = arguments()
    main(args)