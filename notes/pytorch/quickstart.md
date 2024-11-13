https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

# Quickstart
### Data
- DataLoader - iterable around Dataset
	- supports automatic batching, sampling, shuffling, and multiprocess loading
- Dataset - stores the samples and labels

```
train_data = datasets.<data>()

train_loader = DataLoader(train_data, batch_size)

for x,y in train_loader:
```


### Models
- Create a class that inherits from `nn.Module`
- Layers defined in the `__init__` function

### Model Training
Need:
- loss function
- optimizer

