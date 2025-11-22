import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import warnings

warnings.filterwarnings("ignore")


class FFTFeatureExtractor(nn.Module):
    """
    Extracts frequency domain features using FFT.

    Key insight: Time series often have periodic patterns that are easier to
    identify in frequency domain. FFT decomposes the signal into its frequency
    components.
    """

    def __init__(self, n_features: int = 16):
        super().__init__()
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract FFT features from time series

        Args:
            x: Input tensor of shape (batch_size, sequence_length)

        Returns:
            FFT features of shape (batch_size, n_features * 2) for magnitude and phase
        """
        # Compute FFT
        fft = torch.fft.fft(x, dim=-1)

        # Get magnitude and phase
        magnitude = torch.abs(fft)
        phase = torch.angle(fft)

        # Select top-k frequencies (excluding DC component at index 0)
        # We want n_features/2 complex numbers, each giving magnitude and phase
        k = self.n_features // 2

        if not hasattr(self, "_debug"):
            print(f"FFTFeatureExtractor: n_features={self.n_features}, k={k}")
            self._debug = True

        # Get indices of top-k frequencies by magnitude
        # We only look at positive frequencies (first half of FFT)
        positive_freqs = magnitude[:, 1 : x.shape[1] // 2]
        top_k_values, top_k_indices = torch.topk(positive_freqs, k, dim=-1)
        top_k_indices = top_k_indices + 1  # Adjust for DC component offset

        # Gather top-k magnitudes and phases using the values directly
        top_magnitudes = top_k_values

        # Gather phases for the top-k indices
        batch_size = x.shape[0]
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k)
        top_phases = phase.gather(1, top_k_indices)

        # Normalize features
        top_magnitudes = top_magnitudes / (magnitude.sum(dim=1, keepdim=True) + 1e-8)

        # Concatenate magnitude and phase features
        features = torch.cat([top_magnitudes, top_phases], dim=-1)

        # Ensure we have the expected number of features
        assert (
            features.shape[1] == self.n_features
        ), f"Expected {self.n_features} features, got {features.shape[1]}"

        return features


class SimpleTransformerEncoder(nn.Module):
    """
    Simple transformer encoder to simulate Chronos-like behavior.
    In practice, this would be the pre-trained Chronos model.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, n_heads: int = 4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, hidden_dim) * 0.1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_dim, n_heads, dim_feedforward=256, batch_first=True
            ),
            num_layers=2,
        )
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        x = x.unsqueeze(-1)  # Add feature dimension
        x = self.embedding(x)
        x = x + self.positional_encoding[:, : x.shape[1], :]
        x = self.transformer(x)
        # Global average pooling
        x = x.mean(dim=1)
        return self.output_projection(x)


class FFTTransformerFusion(nn.Module):
    """
    Fusion model combining transformer features with FFT features.

    Architecture:
    1. Transformer branch: Captures sequential patterns
    2. FFT branch: Captures frequency patterns
    3. Fusion: Combines both for better predictions
    """

    def __init__(self, n_fft_features: int = 16, hidden_dim: int = 64):
        super().__init__()

        # Transformer for sequential patterns
        self.transformer = SimpleTransformerEncoder(hidden_dim=hidden_dim)

        # FFT feature extractor for frequency patterns
        self.fft_extractor = FFTFeatureExtractor(n_fft_features)

        # Fusion layers
        # Note: n_fft_features is total features (magnitude + phase combined)
        fusion_input_dim = hidden_dim + n_fft_features
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Debug: print dimensions
        print(
            f"Fusion input dimension: {fusion_input_dim} (transformer: {hidden_dim} + FFT: {n_fft_features})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract transformer features
        transformer_features = self.transformer(x)

        # Extract FFT features
        fft_features = self.fft_extractor(x)

        # Debug: check dimensions
        if not hasattr(self, "_debug_printed"):
            print(f"Transformer features shape: {transformer_features.shape}")
            print(f"FFT features shape: {fft_features.shape}")
            self._debug_printed = True

        # Combine features
        combined = torch.cat([transformer_features, fft_features], dim=-1)

        # Generate prediction
        output = self.fusion_layers(combined)

        return output.squeeze()


class BaselineModel(nn.Module):
    """Simple baseline model without FFT features"""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.transformer = SimpleTransformerEncoder(hidden_dim=hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.transformer(x)
        return self.output_layer(features).squeeze()


def generate_synthetic_data(
    n_samples: int = 1000, seq_length: int = 100, n_frequencies: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic time series with multiple frequency components.
    This simulates real-world data with periodic patterns.
    """
    time = np.linspace(0, 10, seq_length + 1)
    X = []
    y = []

    for _ in range(n_samples):
        # Random frequencies (cycles per time unit)
        frequencies = np.random.uniform(0.5, 5, n_frequencies)
        amplitudes = np.random.uniform(0.5, 2, n_frequencies)
        phases = np.random.uniform(0, 2 * np.pi, n_frequencies)

        # Generate signal as sum of sinusoids
        signal = np.zeros(seq_length + 1)
        for freq, amp, phase in zip(frequencies, amplitudes, phases):
            signal += amp * np.sin(2 * np.pi * freq * time + phase)

        # Add noise
        signal += np.random.normal(0, 0.1, seq_length + 1)

        # Use all but last point as input, last point as target
        X.append(signal[:-1])
        y.append(signal[-1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 20,
    lr: float = 1e-3,
) -> Tuple[list, list]:
    """Train the model and return training history"""

    import time
    import sys

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    train_losses = []
    val_losses = []

    print(f"[{time.strftime('%H:%M:%S')}] Starting training for {epochs} epochs...")
    sys.stdout.flush()

    for epoch in range(epochs):
        start_time = time.time()

        # Training
        model.train()
        optimizer.zero_grad()

        predictions = model(X_train)
        loss = criterion(predictions, y_train)

        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = criterion(val_predictions, y_val)

        scheduler.step(val_loss)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        epoch_time = time.time() - start_time

        if (epoch + 1) % 2 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, "
                f"LR: {current_lr:.6f}, Time: {epoch_time:.1f}s"
            )
            sys.stdout.flush()

    return train_losses, val_losses


def evaluate_and_compare(
    baseline_model: nn.Module,
    fft_model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> dict:
    """Compare baseline vs FFT-enhanced model"""

    criterion = nn.MSELoss()
    mae = nn.L1Loss()

    results = {}

    # Evaluate baseline
    baseline_model.eval()
    with torch.no_grad():
        baseline_pred = baseline_model(X_test)
        baseline_mse = criterion(baseline_pred, y_test).item()
        baseline_mae = mae(baseline_pred, y_test).item()

    results["baseline_mse"] = baseline_mse
    results["baseline_mae"] = baseline_mae

    # Evaluate FFT model
    fft_model.eval()
    with torch.no_grad():
        fft_pred = fft_model(X_test)
        fft_mse = criterion(fft_pred, y_test).item()
        fft_mae = mae(fft_pred, y_test).item()

    results["fft_mse"] = fft_mse
    results["fft_mae"] = fft_mae

    # Calculate improvements
    results["mse_improvement"] = (baseline_mse - fft_mse) / baseline_mse * 100
    results["mae_improvement"] = (baseline_mae - fft_mae) / baseline_mae * 100

    return results, baseline_pred, fft_pred


def visualize_results(
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    baseline_pred: torch.Tensor,
    fft_pred: torch.Tensor,
    train_history: dict,
    results: dict,
):
    """Create comprehensive visualizations"""

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))

    # 1. Training curves
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(train_history["baseline_train"], label="Baseline Train", alpha=0.7)
    ax1.plot(train_history["baseline_val"], label="Baseline Val", alpha=0.7)
    ax1.plot(train_history["fft_train"], label="FFT Train", alpha=0.7)
    ax1.plot(train_history["fft_val"], label="FFT Val", alpha=0.7)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training History")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Prediction scatter plot
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(y_test, baseline_pred, alpha=0.5, label="Baseline", s=20)
    ax2.scatter(y_test, fft_pred, alpha=0.5, label="FFT-Enhanced", s=20)
    ax2.plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", alpha=0.5
    )
    ax2.set_xlabel("True Values")
    ax2.set_ylabel("Predictions")
    ax2.set_title("Prediction Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Error distribution
    ax3 = plt.subplot(2, 3, 3)
    baseline_errors = (baseline_pred - y_test).numpy()
    fft_errors = (fft_pred - y_test).numpy()
    ax3.hist(baseline_errors, bins=30, alpha=0.5, label="Baseline", density=True)
    ax3.hist(fft_errors, bins=30, alpha=0.5, label="FFT-Enhanced", density=True)
    ax3.set_xlabel("Prediction Error")
    ax3.set_ylabel("Density")
    ax3.set_title("Error Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Sample predictions
    ax4 = plt.subplot(2, 3, 4)
    sample_idx = 0
    ax4.plot(X_test[sample_idx].numpy(), label="Historical", alpha=0.7)
    ax4.scatter(
        len(X_test[sample_idx]),
        y_test[sample_idx],
        color="green",
        s=100,
        label="True",
        zorder=5,
    )
    ax4.scatter(
        len(X_test[sample_idx]),
        baseline_pred[sample_idx],
        color="blue",
        s=100,
        label="Baseline",
        marker="^",
        zorder=5,
    )
    ax4.scatter(
        len(X_test[sample_idx]),
        fft_pred[sample_idx],
        color="red",
        s=100,
        label="FFT",
        marker="s",
        zorder=5,
    )
    ax4.set_title("Sample Prediction")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. FFT analysis
    ax5 = plt.subplot(2, 3, 5)
    sample_signal = X_test[0].numpy()
    fft_mag = np.abs(np.fft.fft(sample_signal))
    freqs = np.fft.fftfreq(len(sample_signal))
    positive_freqs = freqs[: len(freqs) // 2]
    positive_mags = fft_mag[: len(fft_mag) // 2]
    ax5.plot(positive_freqs[1:], positive_mags[1:])  # Skip DC component
    ax5.set_xlabel("Frequency")
    ax5.set_ylabel("Magnitude")
    ax5.set_title("FFT of Sample Signal")
    ax5.grid(True, alpha=0.3)

    # 6. Performance metrics
    ax6 = plt.subplot(2, 3, 6)
    metrics = ["MSE", "MAE"]
    baseline_vals = [results["baseline_mse"], results["baseline_mae"]]
    fft_vals = [results["fft_mse"], results["fft_mae"]]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax6.bar(x - width / 2, baseline_vals, width, label="Baseline")
    bars2 = ax6.bar(x + width / 2, fft_vals, width, label="FFT-Enhanced")

    ax6.set_ylabel("Error")
    ax6.set_title("Model Comparison")
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax6.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        height = bar.get_height()
        ax6.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig("fft_fusion_results.png", dpi=150)
    plt.close()

    print("\nResults visualization saved to 'fft_fusion_results.png'")


if __name__ == "__main__":
    print("FFT-Enhanced Transformer Fusion - Proof of Concept")
    print("=" * 60)
    print("\nKey Concepts:")
    print("- FFT: Extracts frequency patterns that transformers might miss")
    print("- Fusion: Combines sequential (transformer) and frequency (FFT) features")
    print("- Goal: Demonstrate >10% improvement on synthetic periodic data")
    print("\n" + "=" * 60)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate data
    print("\n1. Generating synthetic multi-frequency time series...")
    X, y = generate_synthetic_data(n_samples=2000, seq_length=100, n_frequencies=3)

    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train = torch.tensor(X[:train_size])
    y_train = torch.tensor(y[:train_size])
    X_val = torch.tensor(X[train_size : train_size + val_size])
    y_val = torch.tensor(y[train_size : train_size + val_size])
    X_test = torch.tensor(X[train_size + val_size :])
    y_test = torch.tensor(y[train_size + val_size :])

    print(
        f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    # Create models
    print("\n2. Creating models...")
    baseline_model = BaselineModel(hidden_dim=64)
    fft_model = FFTTransformerFusion(n_fft_features=16, hidden_dim=64)

    # Count parameters
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    fft_params = sum(p.numel() for p in fft_model.parameters())
    print(f"Baseline model parameters: {baseline_params:,}")
    print(f"FFT fusion model parameters: {fft_params:,}")

    # Train baseline
    print("\n3. Training baseline transformer model...")
    baseline_train_losses, baseline_val_losses = train_model(
        baseline_model, X_train, y_train, X_val, y_val, epochs=20
    )

    # Train FFT model
    print("\n4. Training FFT-enhanced fusion model...")
    fft_train_losses, fft_val_losses = train_model(
        fft_model, X_train, y_train, X_val, y_val, epochs=20
    )

    # Evaluate
    print("\n5. Evaluating models on test set...")
    results, baseline_pred, fft_pred = evaluate_and_compare(
        baseline_model, fft_model, X_test, y_test
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"Baseline MSE: {results['baseline_mse']:.6f}")
    print(f"FFT-Enhanced MSE: {results['fft_mse']:.6f}")
    print(f"MSE Improvement: {results['mse_improvement']:.1f}%")
    print(f"\nBaseline MAE: {results['baseline_mae']:.6f}")
    print(f"FFT-Enhanced MAE: {results['fft_mae']:.6f}")
    print(f"MAE Improvement: {results['mae_improvement']:.1f}%")
    print("=" * 60)

    if results["mse_improvement"] > 10:
        print("\n✅ SUCCESS: Achieved >10% improvement on synthetic data!")
        print("   FFT features successfully enhance transformer predictions.")
    else:
        print("\n⚠️  Improvement <10%, but this is still a positive signal!")

    # Visualize
    print("\n6. Creating visualizations...")
    train_history = {
        "baseline_train": baseline_train_losses,
        "baseline_val": baseline_val_losses,
        "fft_train": fft_train_losses,
        "fft_val": fft_val_losses,
    }
    visualize_results(X_test, y_test, baseline_pred, fft_pred, train_history, results)

    print("\n" + "=" * 60)
    print("Next Steps:")
    print("1. Test on real benchmark dataset (ETTh1)")
    print("2. Try different FFT feature counts")
    print("3. Experiment with fusion strategies")
    print("4. Scale up to larger models")
    print("=" * 60)
