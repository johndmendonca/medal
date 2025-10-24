langs=(chinese french german portuguese spanish)


for lang in "${langs[@]}"; do
    python agents/gpt.py \
        --input_file batches_to_process/pairwise/culture_${lang}.jsonl \
        --type upload \
        --api_key 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 \
        --batched False \
        --provider openrouter
done