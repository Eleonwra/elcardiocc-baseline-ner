from src.dataset import Preprossesing
from src.sentence_splitter import Sentence_splitter
from src.tokenizer import MBERT_Tokenizer


def main():
    file_path = 'data/train_dataset.jsonl' 
    processor = Preprossesing(file_path)

    processor.load_data()
    processor.preprocess()
    processor.show_stats()
    dataset = processor.build_dataset()

    titles = ['ΕΝΗΜΕΡΩΤΙΚΟ ΣΗΜΕΙΩΜΑ',
            'ΣΤΟΙΧΕΙΑ ΑΣΘΕΝΟΥΣ',
            'ΙΣΤΟΡΙΚΟ – ΑΝΤΙΚΕΙΜΕΝΙΚΗ ΕΞΕΤΑΣΗ',
            'ΠΟΡΕΙΑ ΝΟΣΟΥ',
            'ΔΙΑΓΝΩΣΗ ΕΞΟΔΟΥ',
            'ΘΕΡΑΠΕΥΤΙΚΗ ΑΓΩΓΗ - ΕΠΕΜΒΑΣΕΙΣ',
            'ΟΔΗΓΙΕΣ ΚΑΤΑ ΤΗΝ ΕΞΟΔΟ - ΠΑΡΑΤΗΡΗΣΕΙΣ',
            'ΕΡΓΑΣΤΗΡΙΑΚΕΣ ΕΞΕΤΑΣΕΙΣ',
            'ΕΞΕΤΑΣΕΙΣ',
            'Λοιπές εξετάσεις',
            'ΑΙΤΙΑ ΕΙΣΟΔΟΥ – ΑΝΤΙΚΕΙΜΕΝΙΚΗ ΕΞΕΤΕΤΑΣΗ - ΙΣΤΟΡΙΚΟ',
            'ΠΑΡΑΚΛΙΝΙΚΕΣ ΕΞΕΤΑΣΕΙΣ']
    
    splitter = Sentence_splitter(dataset, titles)
    final_data = splitter.split_dataset()
 
    tokenizer = MBERT_Tokenizer(threshold=384)
    tokenized_data = tokenizer.tokenize(final_data)

if __name__ == "__main__":
    main()