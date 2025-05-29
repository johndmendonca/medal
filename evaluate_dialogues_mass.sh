langs=(chinese english french german portuguese spanish)
users=(gemma-3-27b-it gpt-4o-mini)
targets=(aya-expanse-8b aya-expanse-32b Llama-3.1-8B-Instruct Phi-3.5-mini-instruct Qwen2.5-3B-Instruct Qwen2.5-7B-Instruct Llama-3.1-70B-Instruct Qwen2.5-72B-Instruct)
release=RELEASE_CLEANING

evaluator_owner=openai
evaluator=gpt-4.1

for lang in "${langs[@]}"; do
    for user in "${users[@]}"; do
        for target in "${targets[@]}"; do
            dialogue=${release}/dialogues_clean/${lang}/${user}_${target}/${target}/turn-4_${target}
            input=batches_to_process/evaluation/${dialogue}/${evaluator}-${lang}.jsonl
            python tasks/dialogue_evaluation/evaluate_dialogue.py \
                --model ${evaluator} \
                --temperature 0 \
                --dialogue ${dialogue} \
                --lang ${lang}
            echo ${input}
            python agents/gpt.py \
                --input_file ${input} \
                --type upload \
                --api_key $OPENAI_API_KEY \
                --batched False \
                --provider azure
        done
    done
done
