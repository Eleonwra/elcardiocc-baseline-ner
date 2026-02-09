# Standard Library Imports 
import json
import pickle

# Third-Party Libraries 
import pandas as pd
from sklearn.model_selection import train_test_split

# Specialized NLP Libraries 
from datasets import Dataset, Features, Sequence, Value, ClassLabel

class Preprossesing:
    def __init__(self, filepath, test_size=0.2, random_seed=42):
        self.filepath = filepath
        self.test_size = test_size
        self.random_seed = random_seed
        self.raw_data = []
        self.processed_data = []
    
    def load_data(self):
        print(f"Loading data from {self.filepath}...")
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                self.raw_data.append(json.loads(line))
        return self.raw_data
    
    def preprocess(self):
        result_list = []
        for entry in self.raw_data:
            patient_id = entry['id']
            text = entry['text']
            mentions = []
            for label in entry['annotations']:
                label_values = list(label.values())
                mentions.append(label_values)
            result_list.append({
                    'patient_id': patient_id,
                    'label': mentions,
                    'text': text
                })

        result_df = []
        for entry in self.raw_data:
            patient_id = entry['id']
            text = entry['text']
            for annotation in entry['annotations']:
                result_df.append({
                    'patient_id': patient_id,
                    'start': annotation['start'],
                    'end': annotation['end'],
                    'code': annotation['code'],
                    'mention': annotation['mention']
                })
        self.result_df = pd.DataFrame(result_df)
        annotations_new = []
        self.b_entity_count = 0
        self.i_entity_count = 0
        for i, document_data in enumerate(result_list):
            patient_id = document_data["patient_id"]
            text = document_data["text"]
            recognized_entities = []
            previous_end = -1
            for annotation in result_list[i]["label"]:
                start = annotation[0]
                end = annotation[1]
                icd_code = annotation[2]
                text1 = text
                name = text1[start:end]
                words = name.split()
                current_start = start
                for idx, word in enumerate(words):
                    if idx == 0:
                        type = "B-ENTITY"
                        recognized_entities.append({
                        "start": start,
                        "end": start + len(word),
                        "type": type,
                        'word_name': word,
                        'icd_code': icd_code })
                        current_start = start + len(word)
                        self.b_entity_count = self.b_entity_count + 1
                if current_start < end:
                    type = "I-ENTITY"
                    recognized_entities.append({
                    "start": current_start,
                    "end": end,
                    "type": type,
                    'word_name': word,
                    'icd_code': icd_code })
                    self.i_entity_count = self.i_entity_count + 1
                    current_start = current_start + len(word)

            document_entry = {
                "patient_id": patient_id,
                "text": text,
                "recognized_entities": recognized_entities
            }
            annotations_new.append(document_entry)

        tokens_list = []
        ner_tags_list = []
        patient_id_list = []
        for j,dat in enumerate(annotations_new):
            tokens = (list(dat["text"]))
            ner_tags = ["O"] * len(tokens)
            for ent in annotations_new[j]["recognized_entities"]:
                for i in range(ent['start'], ent['end']):
                    ner_tags[i] = ent['type']
            tokens_list.append(tokens)
            ner_tags_list.append(ner_tags)
            patient_id_list.append(dat['patient_id'])

        features = Features({
            "tokens": Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
            "ner_tags": Sequence(feature=ClassLabel(names=["O", "B-ENTITY",'I-ENTITY'], id=None), length=-1, id=None),
            "patient_id": Value(dtype='int64', id=None)
        })
        self.ds = Dataset.from_dict(
            {"tokens": tokens_list, "ner_tags": ner_tags_list,"patient_id": patient_id_list},
            features=features
        )

    def build_dataset(self):
        total_size = len(self.ds)
        train_size = int(total_size * 0.8)
        val_size = total_size-train_size

        print(f"Total Size: {total_size}")
        print(f"Train Size: {train_size}")
        print(f"Validation Size: {val_size}")

        patient_ids = self.ds['patient_id']
        indices = range(len(patient_ids))
        patient_ids_train, patient_ids_val = train_test_split(patient_ids, test_size=val_size, random_state=self.random_seed)

        ds_train_data = [obj for obj in self.ds if obj['patient_id'] in patient_ids_train]
        ds_val_data = [obj for obj in self.ds if obj['patient_id'] in patient_ids_val]

        ds_train = Dataset.from_dict({"tokens": [obj['tokens'] for obj in ds_train_data],
                                    "ner_tags": [obj['ner_tags'] for obj in ds_train_data],
                                    "patient_id": [obj['patient_id'] for obj in ds_train_data]})
        ds_val = Dataset.from_dict({"tokens": [obj['tokens'] for obj in ds_val_data],
                                    "ner_tags": [obj['ner_tags'] for obj in ds_val_data],
                                    "patient_id": [obj['patient_id'] for obj in ds_val_data]})

        self.final_dataset = {'train': ds_train, 'validation': ds_val}
        
        with open('data/final_dataset.pickle', 'wb') as output:
            pickle.dump(self.final_dataset, output)
        return self.final_dataset

    def show_stats(self):
        patient_stats = self.result_df.groupby('patient_id').size().agg(['count', 'min', 'max', 'mean'])

        report = f"""
                    {'='*40}
                          DATASET SUMMARY STATISTICS
                    {'='*40}
                    Total Annotations:      {len(self.result_df):>10}
                    Unique Patients:        {int(patient_stats['count']):>10}

                    Annotations per Patient:
                    - Min:                {patient_stats['min']:>10.2f}
                    - Max:                {patient_stats['max']:>10.2f}
                    - Mean:               {patient_stats['mean']:>10.2f}

                    BIO Entity Counts:
                    - B-ENTITY:           {self.b_entity_count:>10}
                    - I-ENTITY:           {self.i_entity_count:>10}
                    {'='*40}
                    """
        print(report)
        return patient_stats
    