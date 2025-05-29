langs=(chinese english french german portuguese spanish)

evaluator_owner=openai
evaluator=o3


for lang in "${langs[@]}"; do
    dialogue=RELEASE_CLEANING/benchmark_dialogues/${lang}_selected
    python tasks/dialogue_evaluation/evaluate_dialogue.py \
        --temperature 0 \
        --dialogue ${dialogue} \
        --lang ${lang} \
        --model ${evaluator_owner}/${evaluator} 
    
    input=batches_to_process/evaluation/${dialogue}/${evaluator}-${lang}.jsonl
    
    python agents/gpt.py \
        --input_file ${input} \
        --type upload \
        --api_key $OPENAI_KEY \
        --batched False \
        --provider openai
done
