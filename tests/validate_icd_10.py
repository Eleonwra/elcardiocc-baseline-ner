import json
import simple_icd_10 as icd

def load_dataset(input_file):
    dataset = []
    with open(input_file, encoding="utf-8") as json_file:
        for line in json_file:
            record = json.loads(line)
            dataset.append(record)

    return dataset

def check_icd_10_codes(dataset):
    for record in dataset:
        annotations = record['annotations']
        for annotation in annotations:
            code = annotation['code']
            if not icd.is_category(code):
                print(f"Error: Invalid code in dataset (not an ICD-10 category) on ID {record['id']}.")
                return False
    return True

def has_overlapping_intervals(intervals):
    intervals.sort()
    for i in range(len(intervals) - 1):
        current_start, current_end = intervals[i]
        next_start, next_end = intervals[i + 1]
        if current_end > next_start:
            return True
    return False

def check_overlap(dataset):
    for record in dataset:
        annotations = record['annotations']
        pairs = []
        for annotation in annotations:
            start,end = annotation['start'], annotation['end']
            pairs.append((start,end))

        if has_overlapping_intervals(pairs):
            print(f"Overlap check failed for ID {record['id']}: Mentions overlap detected.")
            return False

    return True

input_file = 'data/train_dataset.jsonl'
dataset = load_dataset(input_file)
if check_icd_10_codes(dataset):
    print('ICD-10 codes check OK')
if check_overlap(dataset):
    print('Mentions overlap check OK')