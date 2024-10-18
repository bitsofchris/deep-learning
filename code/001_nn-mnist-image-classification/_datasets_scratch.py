from datasets import load_dataset

ds = load_dataset("ylecun/mnist")
print(ds)

# ds is a DatasetDict
print(ds["train"].info)
print(ds["train"].info.features)

# Load the split we weant
train = ds["train"]
train = load_dataset("ylecun/mnist", split="train")

print(train[0])
