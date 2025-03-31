from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained("PORTULAN/albertina-1b5-portuguese-ptpt-encoder", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("PORTULAN/albertina-1b5-portuguese-ptpt-encoder")
dataset = load_dataset("PORTULAN/glue-ptpt", "rte")

def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="albertina-ptpt-rte",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    fp16=True,
    save_total_limit=2,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()
