from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration


tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained("facebook/rag-token-base")

model_path = "./model_save/"
model = AutoModelForCausalLM.from_pretrained(model_path)

#example
querry = "What is langchain and explain it clearly with example."

#retrieve relevant documents
retrieved_docs = retriever.retrieve(querry)
