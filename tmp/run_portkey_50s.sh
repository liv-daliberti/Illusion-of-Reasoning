PHRASE="Hold on, this reasoning might be wrong. Let's go back and check each step carefully. ||| Actually, this approach doesn't look correct. Let's restart and work through the solution more systematically. ||| Wait, something is not right, we need to reconsider. Let's think this through step by step. ||| Please answer the problem again."

for start in $(seq 0 50 450); do
  python -m src.inference.gateways.providers.portkey \
    --output_dir artifacts/results/gpt4o-math-portkey \
    --model gpt-4o \
    --dataset_id MATH-500 \
    --split test \
    --step 0 \
    --num_samples 8 \
    --temperatures 0 0.05 0.3 0.7 1 \
    --two_pass \
    --second_pass_phrase "$PHRASE" \
    --seed 42 \
    --dataset_start "$start" \
    --num_examples 50 &
done
wait
