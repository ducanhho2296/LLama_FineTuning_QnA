import json
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import GPT2Tokenizer


def load_dataset(file_path):
    questions = []
    answers = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            questions.append(data['questionText'])
            answers.append(data['answers'][0]['answerText'])
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    token_questions = [tokenizer.encode(q, truncation=True, max_length=512) for q in questions]
    token_answers = [tokenizer.encode(a, truncation=True, max_length=512) for a in answers]

    df = pd.DataFrame({'questions': token_questions, 'answers': token_answers})
    return df
                       

def split_dataset(df, test_size=0.1):
    train_df, test_df = train_test_split(df, test_size=test_size)
    return train_df, test_df
    

if __name__ == '__main__':
    file_path = 'test-qar_all.jsonl'
    df = load_dataset(file_path)
    train_df, test_df = split_dataset(df)
