# src/preprocess.py

import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def load_and_prepare_data(config):
    """Loads, flattens, and preprocesses the CUAD dataset."""
    print("--- Starting Data Loading and Preprocessing ---")

    # 1. Load and Flatten the nested JSON data
    print("Loading and flattening the dataset...")
    json_file_path = config["data_path"]
    cuad_dataset = load_dataset('json', data_files=json_file_path, split="train")
    df_nested = pd.DataFrame(cuad_dataset[0]['data'])

    flattened_data = []
    for index, row in df_nested.iterrows():
        contract_title = row['title']
        for para in row['paragraphs']:
            context = para['context']
            for qa in para['qas']:
                record = {'id': qa['id'], 'title': contract_title, 'context': context, 'question': qa['question'], 'answers': qa['answers']}
                flattened_data.append(record)
    final_df = pd.DataFrame(flattened_data)
    print(f"Flattening complete. Created {len(final_df)} examples.")

    # 2. Create Label Mappings
    unique_clauses = final_df['question'].unique().tolist()
    labels_list = ["O"]
    for clause in unique_clauses:
        label_name = clause.replace("What is the ", "").replace("?", "").replace(" ", "_").upper()
        labels_list.append(f"B-{label_name}")
        labels_list.append(f"I-{label_name}")
    label2id = {label: i for i, label in enumerate(labels_list)}
    id2label = {i: label for i, label in enumerate(labels_list)}
    print(f"Created {len(labels_list)} labels.")

    # 3. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])

    # 4. Define and Apply Preprocessing for NER
    def preprocess_for_ner_full(examples):
        # (This is the same robust function we developed)
        tokenized_inputs = tokenizer(examples["context"], max_length=512, truncation=True, stride=128, return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
        sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping"); offset_mapping = tokenized_inputs.pop("offset_mapping")
        labels = []
        for i in range(len(examples["question"])):
            question = examples["question"][i]; answers = examples["answers"][i]
            label_name = question.replace("What is the ", "").replace("?", "").replace(" ", "_").upper()
            b_label_id = label2id[f"B-{label_name}"]; i_label_id = label2id[f"I-{label_name}"]
            if isinstance(answers, dict) and answers.get("answer_start"):
                answer_start = answers["answer_start"][0]; answer_end = answer_start + len(answers["text"][0])
            else:
                answer_start, answer_end = -1, -1
            doc_chunk_indices = [j for j, sample_idx in enumerate(sample_mapping) if sample_idx == i]
            for chunk_idx in doc_chunk_indices:
                chunk_labels = [label2id["O"]] * len(tokenized_inputs["input_ids"][chunk_idx]); chunk_offset_mapping = offset_mapping[chunk_idx]
                token_start_index = 0
                while token_start_index < len(chunk_offset_mapping) and chunk_offset_mapping[token_start_index][0] < answer_start: token_start_index += 1
                token_end_index = token_start_index
                while token_end_index < len(chunk_offset_mapping) and chunk_offset_mapping[token_end_index][1] <= answer_end: token_end_index += 1
                token_end_index -= 1
                if answer_start != -1 and token_start_index <= token_end_index:
                    chunk_labels[token_start_index] = b_label_id
                    for j in range(token_start_index + 1, token_end_index + 1): chunk_labels[j] = i_label_id
                labels.append(chunk_labels)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("Mapping the dataset...")
    full_dataset = Dataset.from_pandas(final_df)
    processed_dataset = full_dataset.map(preprocess_for_ner_full, batched=True, remove_columns=full_dataset.column_names)
    
    print("--- Preprocessing Finished ---")
    return processed_dataset, tokenizer, label2id, id2label