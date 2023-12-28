from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, AutoModelForCausalLM, AutoTokenizer, GPT2ForQuestionAnswering, GPT2Tokenizer
import os
import argparse


def load_gpt_model(path=None):
    if path and os.path.exists(path):
        model = AutoModelForCausalLM.from_pretrained(path)    else:
        #load gpt2 pretrained model from Hugging Face
        model = GPT2ForQuestionAnswering.from_pretrained("gpt2")
    
    return model

model_path = "./model_save/"
model = AutoModelForCausalLM.from_pretrained(model_path)

#example
querry = "What is langchain and explain it clearly with example."

#retrieve relevant documents
retrieved_docs = retriever.retrieve(querry)
