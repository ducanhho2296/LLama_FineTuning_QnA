import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from model import load_model
import pandas as pd

class ChatbotDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

def train(model_name, train_dataset, test_dataset):
    model, tokenizer = load_model(model_name)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    train_dataset = ChatbotDataset(train_dataset.to_dict('list'))
    test_dataset = ChatbotDataset(test_dataset.to_dict('list'))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    training_args = TrainingArguments(
        output_dir='./model_save',
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Same as DataLoader batch size
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True,
        report_to="none"    # Disable logging to WANDB
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader,  
        eval_dataset=test_loader     
    )
    
    # Start the training
    trainer.train()
    
    # Evaluate the model with the test data after training
    trainer.evaluate(eval_dataset=test_loader)

    # Save the fine-tuned model
    model.save_pretrained('./model_save/' + model_name)

if __name__ == "__main__":
    model_name = 'gpt2'  # Or use argparse to set this
    train_df = pd.read_pickle('train_dataset.pkl')
    test_df = pd.read_pickle('test_dataset.pkl')
    
    # Start training
    train(model_name, train_df, test_df)
