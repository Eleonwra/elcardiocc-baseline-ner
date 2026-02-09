from src.dataset import Preprossesing
from src.sentence_splitter import Sentence_splitter
from src.tokenizer import MBERT_Tokenizer

def main():  
    data_path = 'data/final_dataset_sentences.pickle'
    import pickle
    with open(data_path, 'rb') as f:
        final_data = pickle.load(f)

    tokenizer = MBERT_Tokenizer(threshold=384)
    tokenized_data = tokenizer.tokenize(final_data)

if __name__ == "__main__":
    main()