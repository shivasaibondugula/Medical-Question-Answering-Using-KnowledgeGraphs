import json
import os
import spacy
from pathlib import Path
from tqdm import tqdm

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

INPUT_PATH = "data/processed/cleaned_data.json"
OUTPUT_PATH = "data/processed/entities_full_parsed.json"

def extract_features(doc):
    return {
        "lexical": {
            "tokens": [token.text for token in doc],
            "lemmas": [token.lemma_ for token in doc]
        },
        "syntactic": {
            "pos_tags": [(token.text, token.pos_) for token in doc],
            "dependencies": [(token.text, token.dep_, token.head.text) for token in doc]
        },
        "semantic": {
            "entities": [(ent.text, ent.label_) for ent in doc.ents]
        }
    }

def process_data():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"{INPUT_PATH} not found!")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = [record["question"] for record in data]
    answers = [record["answer"] for record in data]

    print("🚀 Running NLP pipeline on questions...")
    q_docs = list(tqdm(nlp.pipe(questions, batch_size=64), total=len(questions), desc="Processing Questions"))

    print("🚀 Running NLP pipeline on answers...")
    a_docs = list(tqdm(nlp.pipe(answers, batch_size=64), total=len(answers), desc="Processing Answers"))

    enriched_data = []
    print("🛠️  Extracting features and combining...")
    for idx, (q_doc, a_doc) in enumerate(tqdm(zip(q_docs, a_docs), total=len(data), desc="Combining Records")):
        enriched_data.append({
            "question": questions[idx],
            "answer": answers[idx],
            "question_nlp": extract_features(q_doc),
            "answer_nlp": extract_features(a_doc)
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ NLP processing complete. Output saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_data()
