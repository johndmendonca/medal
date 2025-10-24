from unsloth import FastLanguageModel
from trl import SFTTrainer,SFTConfig
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
    
    sft_args = HfArgumentParser(SFTConfig).parse_yaml_file(args.args_file)
    
    def preprocess_function(example):
        if example["role"] == "assistant":
            return {
                "prompt": example["context"],
                "completion": [example["response"]],
            }
        return None

    dataset = dataset.map(preprocess_function, remove_columns=["source", "scene", "lang", "model","role"])
    
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Dropout = 0 is currently optimized
        bias="none",  # Bias = "none" is currently optimized
        use_gradient_checkpointing=True,
        random_state=3407,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=sft_args
    )
    trainer.train()


if __name__ == "__main__":
    args = arguments()
    main(args)