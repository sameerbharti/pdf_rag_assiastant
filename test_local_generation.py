from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-small"


# --------------------------------------------------
# 1. Load model
# --------------------------------------------------

print("Loading local model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

print("Model loaded.")


# --------------------------------------------------
# 2. Test prompt
# --------------------------------------------------

prompt = """
Answer the question using the information provided.

Context:
A student can verify their student status by signing in,
going to Edit Profile → Student Details, and clicking
Start Verification.

Question:
How do I verify my student status?
"""


# --------------------------------------------------
# 3. Tokenize
# --------------------------------------------------

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True
)


# --------------------------------------------------
# 4. Generate
# --------------------------------------------------

outputs = model.generate(
    **inputs,
    max_new_tokens=100
)


# --------------------------------------------------
# 5. Decode
# --------------------------------------------------

answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)


# --------------------------------------------------
# 6. Display
# --------------------------------------------------

print("=" * 60)
print("LOCAL GENERATION TEST")
print("=" * 60)

print("\nPrompt:")
print(prompt)

print("\nAnswer:")
print(answer)