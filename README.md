# cricketlm
I am trying to fine tune and cricket a LLM model for Cricket data. I am using LLAMA 3 (meta-llama/Llama-3.1-8B) for the domain training.

**Goal:**

Goal of this experiment to fine tune a model for Cricket domain like an analyst 

**Fine tunning is converted into following pahses:**

- **Data gathering & prepration**:
  - I am using data from https://cricsheet.org. It is an open source data for Cricket for each match with ball by ball details.
  - Cric Sheet provides data in the structured format and I am converting it to the human readable format so, that LLM models can understand the data better and find the relationship.
- **Domain training**:
  - In this phase I am trying to make LLM learn the cricket terminalogies, language and data
- **Tunning**:
  - Trying to make LLM respond like a human not like an AI. Instead of returning only numbers will still be difficult for the humans so, fine tunning the response.     
