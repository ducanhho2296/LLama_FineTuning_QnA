from langchain.chains import SingleLM
from langchain.applications import Chat

# Initialize your model
gpt2_model = MyGPT2Model()

# Use LangChain's SingleLM for conversation
conversation_agent = SingleLM(gpt2_model)

