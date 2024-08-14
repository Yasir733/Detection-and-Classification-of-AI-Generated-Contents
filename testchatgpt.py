import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def is_ai_generated(text):
    # Load pre-trained GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Generate text using the GPT-2 model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

    # Decode and compare the generated text with the original
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # You may adjust this threshold based on your needs
    similarity_threshold = 0.8

    # Check semantic similarity between the input and generated text
    similarity_score = semantic_similarity(text, generated_text)
    
    return similarity_score < similarity_threshold

def semantic_similarity(text1, text2):
    # Implement your semantic similarity function here
    # This could involve using other NLP libraries like spaCy, gensim, or SentenceTransformers
    # For simplicity, a placeholder function is used here
    return 0.5

# Example usage
text_to_check = "Rama Killed Ravana in Ramayana"
result = is_ai_generated(text_to_check)
print(result)
if result:
    print("The text is likely AI-generated.")
else:
    print("The text seems to be human-generated.")
