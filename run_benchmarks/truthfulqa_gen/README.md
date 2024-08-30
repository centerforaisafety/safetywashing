# Truthfulqa

## Running code

### Generate the model outputs

Install and run lm\_eval

```bash
git clone https://github.com/steven-basart/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

We use slurm to launch the jobs and run in parallel across the cluster. 
Feel free to modify the following file to not use slurm for running lm\_eval.

```bash
python run_lm_eval.py
```

### Train gpt4o-mini judge

Add your OpenAI key to an environment variable.

```bash
export OPENAI_API_KEY="your_api_key_here"

# Upload the truth and info files to OpenAI
python upload_files.py


# Modify the next file to add the file ids returned by the previous file.
python train_judge.py
```

### Evaluate the model outputs (with GPT-4o-mini)

```bash
# Modify the next file to add the model ids.
python eval.py
```

Also make sure the file `models.json` exists in `run_benchmarks` folder and points to the models to be run.

