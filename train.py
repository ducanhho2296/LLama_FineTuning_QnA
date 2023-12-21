import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from model import load_model
from torch.utils.data import DataLoader


class ChatbotDataset(Dataset):
    def __init__(self, dataframe):
        self.encodings = dataframe['encodings'].tolist()

    def __getitem__(self, idx):
      item = self.encodings[idx]
      item['labels'] = item['input_ids'].clone().detach().cpu()  # Convert to CPU tensor
      return {key: val.clone().detach().cpu() if isinstance(val, torch.Tensor) else torch.tensor(val) for key, val in item.items()}


    def __len__(self):
        return len(self.encodings)
def train(model_name, train_dataset, test_dataset):
    model, tokenizer = load_model(model_name)
    if torch.cuda.is_available():
      model = model.cuda()
    else:
        print("CUDA is not available. Check your Colab runtime settings.")

    print(next(model.parameters()).device) 
    training_args = TrainingArguments(
        output_dir='./model_save',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    trainer.train()
    model.save_pretrained('./model_save/' + model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2', help='Model name (gpt2, bert)')
    args = parser.parse_args()

    train_df = pd.read_pickle('dataset/train_dataset.pkl')
    test_df = pd.read_pickle('dataset/test_dataset.pkl')
    train_dataset = ChatbotDataset(train_df)
    test_dataset = ChatbotDataset(test_df)

    # Check if GPU is available and select the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the TrainingArguments with the device parameter
    training_args = TrainingArguments(
        output_dir='./model_save',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    model, tokenizer = load_model(args.model)
    model.to(device)  # Move the model to the specified device

    # Move the data in the datasets to the specified device
    train_dataset.encodings = [item.to(device) for item in train_dataset.encodings]
    test_dataset.encodings = [item.to(device) for item in test_dataset.encodings]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()
    model.save_pretrained('./model_save/' + args.model)