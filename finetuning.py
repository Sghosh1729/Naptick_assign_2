import json
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from datasets import Dataset

# Load the dataset
with open("sleep_data_for_finetuning.json", "r") as f:
    qa_pairs = json.load(f)

# Preprocess data into a format that GPT models can use
train_texts = [f"Question: {item['input']}\nAnswer: {item['output']}" for item in qa_pairs]
dataset_dict = {"text": train_texts}
dataset = Dataset.from_dict(dataset_dict)

# Load pre-trained tokenizer and model (distilgpt2 in this case)
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure the tokenizer uses the EOS token for padding
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Tokenization function
def tokenize_function(example):
    tokenized = tokenizer(
        example["text"],
        padding="max_length",  # Pad to the max length
        truncation=True,
        max_length=512,  # You can change max length based on your memory constraints
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()  # Set labels for language modeling
    return tokenized

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Setup the Trainer
training_args = TrainingArguments(
    output_dir="./results",         # Directory to save results
    num_train_epochs=3,             # Number of epochs
    per_device_train_batch_size=1,  # Adjust based on GPU memory
    logging_dir="./logs",           # Directory to save logs
    logging_steps=50,               # Log every 50 steps
    save_steps=500,                 # Save checkpoint every 500 steps
    save_total_limit=2,             # Limit the number of saved checkpoints
    report_to="none",               # Do not report to any service
)

# Define the data collator (for padding sequences dynamically)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_sleep_coach_model")
tokenizer.save_pretrained("./fine_tuned_sleep_coach_tokenizer")

print("Fine-tuning complete and model saved!")







