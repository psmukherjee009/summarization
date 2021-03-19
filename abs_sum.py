#!/usr/bin/env python3

import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def get_summary(text, model_name):
	torch_device = "cuda" if torch.cuda.is_available() else "cpu"
	tokenizer = PegasusTokenizer.from_pretrained(model_name)
	model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

	batch = tokenizer.prepare_seq2seq_batch(text, truncation=True, padding="longest", return_tensors="pt").to(torch_device)

	translated = model.generate(**batch)

	return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

if __name__ == '__main__':
	import sys
	if len(sys.argv) < 2:
		print("Usage", sys.argv[0], "<DocumentFile>")
		sys.exit(1)
	with open(sys.argv[1]) as fh:
		lines = fh.read()
		print(lines)
		print(get_summary(lines, "google/pegasus-xsum"))

