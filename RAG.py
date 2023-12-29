from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, AutoModelForCausalLM, AutoTokenizer, GPT2ForQuestionAnswering, GPT2Tokenizer
import os
import argparse


def load_gpt_model(path=None):
    if path and os.path.exists(path):
        model = AutoModelForCausalLM.from_pretrained(path)    else:
        #load gpt2 pretrained model from Hugging Face
        model = GPT2ForQuestionAnswering.from_pretrained("gpt2")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./model_save/', type=str, help='path to model')
    args = parser.parse_args()
    
    #load generative model
    model = load_gpt_model(args.path)
    print("Model Loaded")

    #retriever
    retriever = RagRetriever.from_pretrained("facebook/rag-token-base")
    #create RAG tokenizer
    rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")

    #example
    querry = "What is langchain and explain it clearly with example."


#retrieve relevant documents
retrieved_docs = retriever.retrieve(querry)
