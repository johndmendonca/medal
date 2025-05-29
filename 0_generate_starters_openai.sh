model=$1
lang=$2
original_dataset=$3

run_id=affective_persona
dataset_path=tasks/narrative_generation/data/${original_dataset}.json

generate() {
    python tasks/narrative_generation/generate_narratives.py \
        --lang ${lang} \
        --model ${model} \
        --type generate \
        --run-id ${run_id} \
        --dataset ${dataset_path}
}

upload() {
    local input_file=$1
    local provider=$2
    local key=$3
    python agents/gpt.py \
        --input_file ${input_file} \
        --type upload \
        --api_key ${key} \
        --provider ${provider}
}


evaluate() {
    python tasks/narrative_generation/generate_narratives.py \
        --lang ${lang} \
        --model ${model} \
        --type evaluate \
        --run-id ${run_id} \
        --dataset ${dataset_path}
}


process() {
    python tasks/narrative_generation/generate_narratives.py \
        --lang ${lang} \
        --model ${model} \
        --type process \
        --run-id ${run_id} \
        --dataset ${dataset_path}
}

# First generation
generate
upload batches_to_process/${run_id}/${original_dataset}_${lang}_turn0_${model}.jsonl openai ${OPENAI_KEY}
echo "Retrieved generations"

# First evaluation
evaluate
upload batches_to_process/${run_id}/${original_dataset}_${lang}_turn0_${model}_eval.jsonl google ${GEMENI_KEY}
echo "Retrieved evaluations"

# Iterative generation until return 0
process
status=$?

echo "Retrieved processed generations."
echo "Regenerating $status examples."

iteration=0
max_iterations=10  # Safety limit to avoid infinite loops

while [ $status -ne 0 ] && [ $iteration -lt $max_iterations ]; do
    upload batches_to_process/${run_id}/${original_dataset}_${lang}_turn0_${model}_regen.jsonl openai ${OPENAI_KEY}
    evaluate
    upload batches_to_process/${run_id}/${original_dataset}_${lang}_turn0_${model}_eval.jsonl google ${GEMENI_KEY}
    process
    status=$?
    
    echo "Regenerating $status examples."
    iteration=$((iteration + 1))
done

if [ $iteration -ge $max_iterations ]; then
    echo "Warning: Reached maximum iterations without completion."
fi


# Process to dialogue dataset
python dialogues/process_batch.py \
    --role user \
    --lang ${lang} \
    --model ${model} \
    --input_batch completed_batches/batches_to_process/${run_id}/${original_dataset}_${lang}_turn0_${model}.jsonl \
    --source batches_to_process/${run_id}/${original_dataset}_${lang}_turn0_${model}.jsonl
