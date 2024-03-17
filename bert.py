from transformers import BertTokenizer, BertModel, pipeline
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# model = BertModel.from_pretrained("bert-base-cased")
# text = "Hows the weather up there"
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)

unmasker = pipeline('fill-mask', model='bert-base-cased')
res = unmasker("TU Delft is a [MASK] university in Delft.")
print(res)
