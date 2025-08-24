
Expert in data loading pytorch.
[[Ray]]

# Ray Data

dataset -> blocks
batches of a block apply udf to
give dataset to ray train - dataset here should be 1 row is a sample
each worker gets shard (many blocks)


- Build **Dataset** - read data into a dataset object (**blocks**), built by driver
	- can randomize with shuffle="files"
	- dataset object (with map_batches) - takes data and yields sample
	- override_num_blocks - specify num of read tasks
- Transform Data
	- map_batches - apply udf to your data
		- data is split into blocks - your udf is called on batches from each block
		- one block is processed by one task
		- never crosses block boundaries
	- map_groups()
- Feed Data to Ray Train
	- TorchTrainer(datasets="train": train_dataset)
	-  get shard ()
		- a shard is a stream of blocks - each worker gets one shard
	- in the train_loop_per_worker() - the entry point for each Ray Train worker
		- inside train loop - for batch in shard.iter_torch_batches()
			- this func replaces Pytorch Dataloaders
			- how much the model gets each step - the number of samples to take from the shard
			- can do a local shuffle here -  local_shuffle_buffer_size num rows to shuffle
		- dataloader / iter_batches - takes dataset and yields batches of samples

```
# Driver (head node)
ds = ray.data.read_parquet("s3://...")           # or read_iceberg(...)
samples = (
    ds
    # (optional) groupby(...).map_groups(make_windows)  # if you need 32-step windows
    .map_batches(transform_fn, batch_format="pandas")   # produce 1 row = 1 sample
    .random_shuffle(seed=42)
)

trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    datasets={"train": samples},   # <-- pass via dict
    scaling_config={"num_workers": N, "use_gpu": True},
)
result = trainer.fit()


# Worker code
def train_loop_per_worker(config):
    shard = train.get_dataset_shard("train")  # this worker's slice of the dataset
    # Yield minibatches of N samples (rows) each:
    for batch in shard.iter_torch_batches(
        batch_size=128,                       # training batch size
        dtypes={"X": torch.float32, "y": torch.long},
        prefetch_batches=2
    ):
        X = batch["X"].to(train.torch.get_device(), non_blocking=True)
        y = batch["y"].to(train.torch.get_device(), non_blocking=True)
        # forward/backward/opt.step()


```

https://docs.ray.io/en/latest/data/working-with-pytorch.html#pytorch-dataloader