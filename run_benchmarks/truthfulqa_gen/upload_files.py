# info
from openai import OpenAI

client = OpenAI()

# upload file
client.files.create(
    file=open("data/finetune_truth.jsonl", "rb"),
    purpose="fine-tune"
)

client.files.create(
    file=open("data/finetune_info.jsonl", "rb"),
    purpose="fine-tune"
)

