# src/train.py

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from preprocess import load_and_prepare_data

def main():
    """Main function to run the training pipeline."""
    
    # 1. Configuration
    # All key parameters are defined here for easy access
    config = {
        "data_path": "data\CUAD_v1\CUAD_v1.json",
        "model_checkpoint": "distilbert-base-uncased",
        "output_dir": "./legal-ner-model-full",
        "learning_rate": 2e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "weight_decay": 0.01
    }

    # 2. Load and Preprocess Data
    processed_dataset, tokenizer, label2id, id2label = load_and_prepare_data(config)

    # 3. Load Model
    model = AutoModelForTokenClassification.from_pretrained(
        config["model_checkpoint"], 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # 4. Define Training Arguments (using the legacy version that works for you)
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        evaluate_during_training=True,
        logging_steps=500,
        save_steps=500,
        fp16=True # For GPU speedup
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        eval_dataset=processed_dataset, # Using full dataset for evaluation
        tokenizer=tokenizer,
    )

    # 6. Start Training
    print("\n--- Starting Model Training ---")
    trainer.train()
    print("--- Training Finished ---")

    # 7. Save the final model
    trainer.save_model(config["output_dir"])
    print(f"Final model saved to {config['output_dir']}")

if __name__ == "__main__":
    main()