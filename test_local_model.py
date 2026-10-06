from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-small"


print("=" * 60)
print("LOCAL MODEL DOWNLOAD TEST")
print("=" * 60)

print(f"Model: {MODEL_NAME}")
print("\nDownloading/loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded.")

print("\nDownloading/loading model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

print("Model loaded successfully.")

print("\n" + "=" * 60)
print("MODEL TEST COMPLETE")
print("=" * 60)

print(f"Tokenizer type: {type(tokenizer).__name__}")
print(f"Model type: {type(model).__name__}")