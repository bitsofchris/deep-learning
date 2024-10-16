# 2024-10-16 - training data mnist

### Hugging Face
Decided to use the `datasets` library from Hugging Face to get the data.

Loaded it from [here](https://huggingface.co/datasets/ylecun/mnist).

This was a good opportunity to get more familiar with Hugging Face.

### Pre-Process Pattern with Hugging Face Datasets
Here's what I've got so far:

1. Load the dataset- split into train, validation, test. Optionally, explore.
2. Write your `pre_process` function that will be applied to each sample
3. Call `<dataset>.map(pre_process)` to map your pre-process function to every sample in the dataset.
4. Go train models.

### Training MNIST data
I didn't inspect the data too much - other than checking the shape and that it looked correct. Then I saved to numpy arrays so I don't need to spend 2 minutes downloading and pre-processing the data each run.

However, running it now (yes I am using the test data set for eval, I know this is incorrect), my model doesn't seem to be learning.

### Next Steps
- [ ] debug why my NN is not learning - probably should start with the data and ensure it's noramlized and labeled correctly, but then also that its the right structure for my network
	- [ ] maybe a first step is to create the list of (x),(y) in my loader class and save that rather than 4 separate arrays/ files. use that to do some inspecting on
- [ ] play with my network, try to get it to 96% on the validation set (I think this is common)
- [ ] parallelize this with ray - implement some profiling to see the performance