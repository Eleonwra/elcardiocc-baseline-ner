import pickle
import datasets
import torch
from transformers import BertTokenizerFast

class MBERT_Tokenizer:
    def __init__(self, model_name="bert-base-multilingual-cased", threshold=384):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.threshold = threshold

    def tokenize_section(self, data_section):
        texts = ["".join(t) for t in data_section["sentences_tokens"]]
        tokenized_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.threshold)
        labels = []
        for row_idx, label_old in enumerate(data_section["sentences_ner_tags"]):
            label_new = [[] for t in tokenized_inputs.tokens(batch_index=row_idx)]
            for char_idx in range(len(data_section["sentences_tokens"][row_idx])):
                token_idx = tokenized_inputs.char_to_token(row_idx, char_idx)
                if token_idx is not None:
                    label_new[token_idx].append(data_section["sentences_ner_tags"][row_idx][char_idx])
            
            label_new = list(map(lambda i : max(i, default=0), label_new))
            labels.append(label_new)
        tokenized_inputs["labels"] = labels + [0] * (self.threshold - len(labels))
        return tokenized_inputs

    def tokenize(self, dataset):
        tokenized_data = {'train': [], 'validation': []}
        for section_name, section_data in dataset.items():
            for i in range(len(section_data)):
                tokenized_document = self.tokenize_section(section_data[i])
                tokenized_data[section_name].append(tokenized_document)
        
        with open('data/mBERT_tokenized.pkl', 'wb') as f:
            pickle.dump(tokenized_data, f)
        return tokenized_data
