from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "batterydata/batterybert-cased"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Optionally save them locally
save_dir = "../models/batterybert-cased"
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
