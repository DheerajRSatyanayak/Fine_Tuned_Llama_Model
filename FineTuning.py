
# in bash/terminal:

!pip install -U "transformers>=4.53.0"

# Code:
import transformers
print(transformers.__version__)
from huggingface_hub import login
login()
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B-Instruct"  # Change if you use a different model!

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")  # or "cpu" if you hit GPU issues
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# in bash/terminal
!rm -rf ~/.cache/huggingface/datasets
!pip install -U datasets

# Code:
from datasets import load_dataset

dataset = load_dataset("csv", data_files="./sarcasm.csv", split="train")
def apply_chat_template(example):
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt}

dataset = dataset.map(apply_chat_template)
dataset = dataset.train_test_split(test_size=0.05)

def tokenize_fn(example):
    toks = tokenizer(example["prompt"], padding="max_length", truncation=True, max_length=128)
    toks["labels"] = [
        -100 if t == tokenizer.pad_token_id else t for t in toks["input_ids"]
    ]
    return toks

tokenized = dataset.map(tokenize_fn)
tokenized = tokenized.remove_columns(["question", "answer", "prompt"])

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=40,
    logging_steps=40,
    save_steps=150,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    fp16=False,     # <-- Set this to False
    bf16=False,     # <-- Add this line if not present
    report_to="none",
    learning_rate=1e-5,
    max_grad_norm=2,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")

from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="./fine-tuned-model",  # NOT meta-llama/...
    tokenizer="./fine-tuned-model",
    device_map="auto"
)

# give your prompt here
messages = [{"role": "user", "content": "Prompt"}]
outputs = pipe(messages, max_new_tokens=128)
print(outputs[0]["generated_text"])