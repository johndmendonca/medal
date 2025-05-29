langs=(chinese english french german portuguese spanish)

evaluator_owner=openai
evaluator=gpt-4o-2024-11-20

for lang in "${langs[@]}"; do
    dialogue=RELEASE_CLEANING/sampled_dialogues/${lang}
    python tasks/dialogue_evaluation/evaluate_dialogue.py \
        --model ${evaluator_owner}/${evaluator} \
        --temperature 0 \
        --dialogue ${dialogue} \
        --lang ${lang}
    
    input=batches_to_process/evaluation/${dialogue}/${evaluator}-${lang}.jsonl
    
    python agents/gpt.py \
        --input_file ${input} \
        --type upload \
        --api_key $OPENROUTER_KEY \
        --batched False \
        --provider openrouter
done