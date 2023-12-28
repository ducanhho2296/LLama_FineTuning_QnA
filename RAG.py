from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration


tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained("facebook/rag-token-base")
