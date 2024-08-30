
# info
from openai import OpenAI

client = OpenAI()

# specifiying the uploaded file from logs from the upload job above

uploaded_truth_file = ""
uploaded_info_file = ""

assert uploaded_truth_file and uploaded_info_file

client.fine_tuning.jobs.create(
    training_file=uploaded_truth_file, 
    model="gpt-4o-mini-2024-07-18",
     hyperparameters=dict(batch_size=21, n_epochs=5)
)

client.fine_tuning.jobs.create(
    training_file=uploaded_info_file, 
    model="gpt-4o-mini-2024-07-18",
    hyperparameters=dict(batch_size=21, n_epochs=5)
)

# print out jobs and jobs status
for job in client.fine_tuning.jobs.list(limit=10):
    print(job)
    print("====")
