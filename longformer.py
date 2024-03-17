from transformers import AutoTokenizer, LongformerForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")

TXT = "My friends are <mask> but they eat too many carbs. That's why I decide not to eat with them."

input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
logits = model(input_ids).logits

masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
probs = logits[0, masked_index].softmax(dim=0)
values, predictions = probs.topk(5)

res = tokenizer.decode(predictions).split()
print(res)
