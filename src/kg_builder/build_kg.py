import json
import ijson
from neo4j import GraphDatabase

# Configure Neo4j connection
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Shiva@123$"  # Replace with your actual password

INPUT_FILE = "data/processed/entities_full_parsed.json"

class MedicalKGBuilder:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✅ Neo4j graph cleared.")

    def close(self):
        self.driver.close()

    def create_entity(self, label, name):
        with self.driver.session() as session:
            session.run(
                f"MERGE (n:{label} {{name: $name}})", name=name
            )

    def create_relationship(self, src_label, src_name, rel, tgt_label, tgt_name):
        with self.driver.session() as session:
            session.run(
                f"""
                MERGE (a:{src_label} {{name: $src}})
                MERGE (b:{tgt_label} {{name: $tgt}})
                MERGE (a)-[:{rel}]->(b)
                """,
                src=src_name,
                tgt=tgt_name
            )

    def build_kg(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            records = ijson.items(f, "item")

            for idx, record in enumerate(records):
                disease = None
                # Find disease entity in question
                for ent, label in record["question_nlp"]["semantic"]["entities"]:
                    if "disease" in ent.lower() or "tumor" in ent.lower():
                        disease = ent
                        self.create_entity("Disease", disease)
                        break

                if not disease:
                    continue  # Skip if no valid disease found

                for ent, label in record["answer_nlp"]["semantic"]["entities"]:
                    entity = ent.strip()
                    if not entity or len(entity) < 2:
                        continue

                        # 1. Assume symptoms
                    if any(keyword in entity.lower() for keyword in ["pain", "fever", "seizure", "headache", "fatigue"]):
                        self.create_entity("Symptom", entity)
                        self.create_relationship("Disease", disease, "HAS_SYMPTOM", "Symptom", entity)

                        # 2. Tests or investigations
                    elif any(entity.lower() in s for s in ["mri", "ct", "scan", "x-ray", "ecg", "eeg"]):
                        self.create_entity("Test", entity)
                        self.create_relationship("Disease", disease, "REQUIRES_TEST", "Test", entity)

                        # 3. Treatments
                    elif any(keyword in entity.lower() for keyword in
                             ["therapy", "drug", "treatment", "surgery", "medication"]):
                        self.create_entity("Treatment", entity)
                        self.create_relationship("Disease", disease, "TREATED_BY", "Treatment", entity)

                        # 4. Recommendations
                    elif any(keyword in entity.lower() for keyword in ["see", "consult", "urgent", "emergency", "visit"]):
                        self.create_entity("Action", entity)
                        self.create_relationship("Disease", disease, "RECOMMENDED_ACTION", "Action", entity)

                        # 5. Specialists
                    elif any(keyword in entity.lower() for keyword in ["ologist", "surgeon", "specialist", "doctor"]):
                        self.create_entity("Specialist", entity)
                        self.create_relationship("Disease", disease, "SPECIALIST_FOR", "Specialist", entity)

                    else:
                        # Fallback as generic related entity
                        self.create_entity("Related", entity)
                        self.create_relationship("Disease", disease, "RELATED_TO", "Related", entity)

                if idx % 1000 == 0:
                    print(f"Processed {idx} records...")

        print("✅ Knowledge Graph built successfully in Neo4j.")

if __name__ == "__main__":
    builder = MedicalKGBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    builder.build_kg(INPUT_FILE)
    builder.close()
