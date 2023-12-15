import argparse
import pandas as pd
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from prepare_dataset import load_dataset, split_dataset
from model import load_model


class ChatbotDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

def train(model_name, train_dataset, test_dataset):
    model = GPT2LMHeadModel.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir='./model_save',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    model.save_pretrained('./model_save/' + model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2', help='Model name (gpt2, bert)')
    args = parser.parse_args()

    # Load preprocessed datasets
    file_path = 'test-qar_all.jsonl'
    df = load_dataset(file_path)
    train_df, test_df = split_dataset(df)

    # Convert dataframes to datasets
    train_dataset = ChatbotDataset(train_df)
    test_dataset = ChatbotDataset(test_df)

    train(args.model, train_dataset, test_dataset)
