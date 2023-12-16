import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer
from concurrent.futures import ProcessPoolExecutor

def tokenize_data(pair):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return tokenizer.encode_plus(pair[0], pair[1], truncation=True, max_length=512, padding='max_length', return_tensors='pt')

def load_dataset(file_path):
    pairs = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            question = data['questionText']
            answer = data['answers'][0]['answerText']
            pairs.append((question, answer))

    # Parallel tokenization
    with ProcessPoolExecutor() as executor:
        tokenized_pairs = list(executor.map(tokenize_data, pairs))

    return pd.DataFrame({'encodings': tokenized_pairs})

def split_dataset(df, test_size=0.1):
    train_df, test_df = train_test_split(df, test_size=test_size)
    return train_df, test_df

if __name__ == '__main__':
    file_path = 'test-qar_all.jsonl'
    df = load_dataset(file_path)
    train_df, test_df = split_dataset(df)
