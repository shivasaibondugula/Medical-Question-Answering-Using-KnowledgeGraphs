import os
import json
from datasets import load_dataset
from pathlib import Path


def load_and_preprocess_dataset(dataset_name="Malikeh1375/medical-question-answering-datasets", config="all-processed", split="train"):
    print(f"Loading dataset '{dataset_name}' with config '{config}'...")
    dataset = load_dataset(dataset_name, config, split=split)

    print("Preprocessing dataset...")
    processed_data = []

    for record in dataset:
        question = record.get("input", "").strip()
        answer = record.get("output", "").strip()

        if question and answer:
            processed_data.append({
                "question": question,
                "answer": answer
            })

    print(f"Processed {len(processed_data)} records.")
    return processed_data


def save_processed_data(processed_data, output_path="data/processed/cleaned_data.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    print(f"Saved cleaned data to {output_path}")


if __name__ == "__main__":
    data = load_and_preprocess_dataset()
    save_processed_data(data)
