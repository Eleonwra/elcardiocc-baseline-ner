import pickle
import re
import stanza
import os
from datasets import Features, Sequence, Value, ClassLabel, Dataset

class Sentence_splitter:
    def __init__(self, data, titles):
            self.titles = titles
            self.data = data
            stanza.download('el', processors='tokenize')
            self.nlp_stanza = stanza.Pipeline('el')

    def split_text_by_titles(self, text):
        title_pattern = re.compile('|'.join([title for title in self.titles]))
        start_title_positions = [match.start() for match in re.finditer(title_pattern, text)]
        title_positions = start_title_positions + [len(text)]
        sections = [text[title_positions[i]:title_positions[i+1]] for i in range(len(title_positions)-1)]
        return sections, title_positions
    
    def tokenize_text_with_indices(self, text):
        doc = self.nlp_stanza(text)
        tokenized_sentences = []
        for sentence in doc.sentences:
            pattern = re.escape(sentence.text)
            match = re.search(pattern, text)
            if match:
                start_index = match.start()
                end_index = match.end()
                sentence_info = {
                    'text': sentence.text,
                    'start_index': start_index,
                    'end_index': end_index
                }
                tokenized_sentences.append(sentence_info)
        return tokenized_sentences
    
    def split_dataset(self):
        for split in self.data.keys():
            sections_tokens_list = []
            sections_ner_tags_list = []
            patient_id_list = []
            for example in self.data[split]:
                character_tokens = example['tokens']
                text = "".join(character_tokens)
                sections, title_positions =  self.split_text_by_titles(text)
                sections_tokens_list.append([list(section) for section in sections])
                sections_ner_tags = []
                ner_tags = example['ner_tags']
                for i, section in enumerate(sections):
                    start = title_positions[i]
                    end = title_positions[i + 1]
                    sections_ner_tags.append(ner_tags[start:end])
                sections_ner_tags_list.append(sections_ner_tags)
                patient_id_list.append(example["patient_id"])

            features=Features({
                "patient_id": Value(dtype='int32', id=None),
                "sections_tokens": Sequence(feature=Sequence(feature=Value(dtype='string', id=None), length=-1, id=None), length=-1, id=None),
                "sections_ner_tags": Sequence(feature=Sequence(feature=ClassLabel(names=["O", "B-ENTITY",'I-ENTITY'], id=None), length=-1, id=None), length=-1, id=None),
            })
            self.data[split] = Dataset.from_dict(
            {'sections_tokens': sections_tokens_list,'sections_ner_tags':sections_ner_tags_list, "patient_id":  patient_id_list},features=features)

        with open('data/final_dataset_sections.pickle', 'wb') as output:
            pickle.dump(self.data, output)
            
        for split in self.data.keys():
            sentences_tokens_list = []
            sentences_ner_tags_list = []
            patient_id_list = []
            extra_info_list = []
            for j, example in enumerate(self.data[split]):
                if j % 10 == 0: 
                    print(f"  > Processing patient {j} of {len(self.data[split])}...")
                doc_sentences_ner_tags_list = []
                doc_patient_id_list = []
                doc_extra_info_list = []
                doc_sentences_tokens_list = []
                for i in range(len(example['sections_tokens'])):
                    character_tokens = example['sections_tokens'][i]
                    text = "".join(character_tokens)
                    sentences_info =  self.tokenize_text_with_indices(text)
                    sentences = [info['text'] for info in sentences_info]
                    doc_sentences_tokens_list.append([list(sentence) for sentence in sentences])
                    sentences_ner_tags = []
                    ner_tags = example['sections_ner_tags'][i]

                    for info in sentences_info:
                        start_index = info['start_index']
                        end_index = info['end_index']
                        sentence_ner_tags = ner_tags[start_index:end_index]
                        sentences_ner_tags.append(sentence_ner_tags)
                    doc_sentences_ner_tags_list.append(sentences_ner_tags)

                doc_sentences_tokens_list = [item for sublist in doc_sentences_tokens_list for item in sublist]
                doc_sentences_ner_tags_list = [item for sublist in doc_sentences_ner_tags_list for item in sublist]

                sentences_tokens_list.append(doc_sentences_tokens_list)
                sentences_ner_tags_list.append(doc_sentences_ner_tags_list)
                patient_id_list.append(example["patient_id"])
            features=Features({
                "patient_id": Value(dtype='int32', id=None),
                "sentences_tokens": Sequence(feature=Sequence(feature=Value(dtype='string', id=None), length=-1, id=None), length=-1, id=None),
                "sentences_ner_tags": Sequence(feature=Sequence(feature=ClassLabel(names=["O", "B-ENTITY",'I-ENTITY'], id=None), length=-1, id=None), length=-1, id=None),
            })
            self.data[split] = Dataset.from_dict(
            {'sentences_tokens': sentences_tokens_list,'sentences_ner_tags': sentences_ner_tags_list, "patient_id":  patient_id_list},features=features)

        with open('data/final_dataset_sentences.pickle', 'wb') as output:
            pickle.dump(self.data, output)
        return self.data