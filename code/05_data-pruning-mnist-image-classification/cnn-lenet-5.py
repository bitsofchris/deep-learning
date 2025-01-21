import time
import torch
import torch.nn as nn

from data_utils import get_data_loaders
from model import LeNet5


def train_and_evaluate(
    pruning_method,
    dataset_size,
    use_pca=False,
    batch_size=64,
    learning_rate=0.001,
    num_epochs=10,
    runs=3,
):
    """
    Runs multiple training loops for a given pruning_method + dataset_size + optional PCA,
    returns best accuracy across all runs and time taken.
    """

    best_acc = 0.0
    best_time = 0.0
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # We might do multiple runs to see if there's variance in training
    for run_id in range(runs):
        print(
            f"--- {pruning_method} | size={dataset_size} | PCA={use_pca} | run={run_id+1}/{runs} ---"
        )

        # 1. Build pruning_kwargs
        if pruning_method == "random":
            # E.g., "percentage" approach
            # If full MNIST is 60k, then percentage = dataset_size / 60000
            percentage = dataset_size / 60000.0
            pruning_kwargs = {"percentage": percentage}
        elif pruning_method == "cluster":
            # K = desired final size
            pruning_kwargs = {
                "k": dataset_size,
                "use_pca": False,
            }  # we'll handle the PCA variant separately
        elif pruning_method == "cluster_pca":
            # same as cluster, but we also pass a flag to do PCA
            pruning_kwargs = {"k": dataset_size, "use_pca": True}
        else:
            pruning_kwargs = {}

        # 2. Create data loaders
        start_time = time.time()
        train_loader, test_loader = get_data_loaders(
            batch_size=batch_size,
            pruning_method=pruning_method,
            pruning_kwargs=pruning_kwargs,
        )

        # 3. Define model, loss, optimizer
        model = LeNet5(num_classes=10).to(device)
        cost = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 4. Train
        # total_step = len(train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = cost(outputs, labels)

                # Backward & optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if (i + 1) % 400 == 0:
                #     print(
                #         f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}"
                #     )

        # 5. Test
        with torch.no_grad():
            model.eval()
            correct, total = 0, 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = 100.0 * correct / total

        run_time = time.time() - start_time
        print(f"Run={run_id+1}: accuracy={acc:.2f}%, time={run_time:.2f} sec\n")

        # Track best accuracy
        if acc > best_acc:
            best_acc = acc
            best_time = run_time

    return best_acc, best_time


if __name__ == "__main__":
    # pruning_methods = ["random", "cluster", "cluster_pca"]
    # dataset_sizes = [600, 1200, 3000, 6000, 12000]
    pruning_methods = ["random", "cluster"]
    dataset_sizes = [
        600,
        1200,
    ]

    results = []

    for method in pruning_methods:
        for size in dataset_sizes:
            best_acc, best_time = train_and_evaluate(
                pruning_method=method,
                dataset_size=size,
                use_pca=(method == "cluster_pca"),
                batch_size=64,
                learning_rate=0.001,
                num_epochs=5,
                runs=3,  # do 3 runs, pick best
            )
            results.append(
                {
                    "method": method,
                    "size": size,
                    "best_acc": best_acc,
                    "best_time": best_time,
                }
            )
    print("--- FINAL RESULTS ---")
    for r in results:
        print(r)
