import json
import subprocess

with open("../models.json", "r") as f:
    data = json.load(f)

for model in data:
    model_name = data[model]["local_path"]
    run_command = (f"srun lm_eval --tasks=truthfulqa_gen " +
f"--model=hf --model_args pretrained={model_name}  --output_path=results " +
f"--device=cuda:0 --batch_size=1 --log_samples")
    print(run_command)
    subprocess.run(run_command.split())

