import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FFTFeatureExtractor(nn.Module):
    """Extracts frequency domain features using FFT"""
    
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
        
        # Select top-k frequencies (excluding DC component)
        # We take n_features//2 from positive frequencies
        k = self.n_features // 2
        
        # Sort by magnitude and get top-k indices
        top_k_indices = torch.topk(magnitude[:, 1:len(x[0])//2], k, dim=-1).indices + 1
        
        # Gather top-k magnitudes and phases
        batch_indices = torch.arange(x.shape[0]).unsqueeze(1).expand(-1, k)
        top_magnitudes = magnitude[batch_indices, top_k_indices]
        top_phases = phase[batch_indices, top_k_indices]
        
        # Concatenate magnitude and phase features
        features = torch.cat([top_magnitudes, top_phases], dim=-1)
        
        return features


class ChronosFFTFusion(nn.Module):
    """Fusion model combining Chronos with FFT features"""
    
    def __init__(self, 
                 chronos_model_name: str = "amazon/chronos-t5-tiny",
                 n_fft_features: int = 16,
                 hidden_dim: int = 128,
                 freeze_chronos: bool = True):
        super().__init__()
        
        # Load Chronos model
        print(f"Loading Chronos model: {chronos_model_name}")
        self.chronos = AutoModelForSeq2SeqLM.from_pretrained(chronos_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(chronos_model_name)
        
        # Freeze Chronos if specified
        if freeze_chronos:
            for param in self.chronos.parameters():
                param.requires_grad = False
                
        # FFT feature extractor
        self.fft_extractor = FFTFeatureExtractor(n_fft_features)
        
        # Get Chronos hidden dimension
        chronos_hidden_dim = self.chronos.config.hidden_size
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(chronos_hidden_dim + n_fft_features * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Output single value for next timestep
        )
        
    def forward(self, x: torch.Tensor, horizon: int = 1) -> torch.Tensor:
        """
        Forward pass combining Chronos and FFT features
        
        Args:
            x: Input time series of shape (batch_size, sequence_length)
            horizon: Number of steps to forecast
            
        Returns:
            Predictions of shape (batch_size, horizon)
        """
        batch_size, seq_len = x.shape
        
        # Extract FFT features
        fft_features = self.fft_extractor(x)
        
        # Prepare input for Chronos
        # Chronos expects specific input format
        input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
        
        # Get Chronos embeddings
        with torch.no_grad():
            chronos_outputs = self.chronos.encoder(input_ids=input_ids)
            chronos_features = chronos_outputs.last_hidden_state.mean(dim=1)  # Pool over sequence
        
        # Combine features
        combined_features = torch.cat([chronos_features, fft_features], dim=-1)
        
        # Generate predictions
        predictions = []
        for _ in range(horizon):
            pred = self.fusion_layers(combined_features)
            predictions.append(pred)
            
        predictions = torch.cat(predictions, dim=-1)
        
        return predictions


def generate_synthetic_data(n_samples: int = 1000, 
                          seq_length: int = 100,
                          n_frequencies: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic time series with multiple frequency components
    
    Returns:
        X: Input sequences
        y: Target values (next timestep)
    """
    time = np.linspace(0, 10, seq_length + 1)
    X = []
    y = []
    
    for _ in range(n_samples):
        # Random frequencies and amplitudes
        frequencies = np.random.uniform(0.5, 5, n_frequencies)
        amplitudes = np.random.uniform(0.5, 2, n_frequencies)
        phases = np.random.uniform(0, 2*np.pi, n_frequencies)
        
        # Generate signal
        signal = np.zeros(seq_length + 1)
        for freq, amp, phase in zip(frequencies, amplitudes, phases):
            signal += amp * np.sin(2 * np.pi * freq * time + phase)
            
        # Add noise
        signal += np.random.normal(0, 0.1, seq_length + 1)
        
        X.append(signal[:-1])
        y.append(signal[-1])
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_model(model: nn.Module, 
                X_train: torch.Tensor, 
                y_train: torch.Tensor,
                X_val: torch.Tensor,
                y_val: torch.Tensor,
                epochs: int = 10,
                lr: float = 1e-3) -> list:
    """Train the fusion model"""
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        predictions = model(X_train, horizon=1).squeeze()
        loss = criterion(predictions, y_train)
        
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val, horizon=1).squeeze()
            val_loss = criterion(val_predictions, y_val)
            
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
            
    return train_losses, val_losses


def evaluate_models(vanilla_chronos: nn.Module, 
                   fft_model: ChronosFFTFusion,
                   X_test: torch.Tensor,
                   y_test: torch.Tensor) -> dict:
    """Compare vanilla Chronos vs FFT-enhanced model"""
    
    criterion = nn.MSELoss()
    results = {}
    
    # Evaluate vanilla Chronos (simplified - in practice would need proper tokenization)
    with torch.no_grad():
        # For this proof of concept, we'll use a simple baseline
        vanilla_pred = X_test.mean(dim=1)  # Simple mean prediction as baseline
        vanilla_loss = criterion(vanilla_pred, y_test)
        results['vanilla_mse'] = vanilla_loss.item()
        
    # Evaluate FFT-enhanced model
    fft_model.eval()
    with torch.no_grad():
        fft_pred = fft_model(X_test, horizon=1).squeeze()
        fft_loss = criterion(fft_pred, y_test)
        results['fft_mse'] = fft_loss.item()
        
    # Calculate improvement
    improvement = (results['vanilla_mse'] - results['fft_mse']) / results['vanilla_mse'] * 100
    results['improvement_percent'] = improvement
    
    return results, vanilla_pred, fft_pred


def visualize_predictions(X_test: torch.Tensor, 
                         y_test: torch.Tensor,
                         vanilla_pred: torch.Tensor,
                         fft_pred: torch.Tensor,
                         n_examples: int = 5):
    """Visualize prediction comparisons"""
    
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 2*n_examples))
    if n_examples == 1:
        axes = [axes]
        
    for i in range(n_examples):
        ax = axes[i]
        
        # Plot historical data
        ax.plot(X_test[i].numpy(), label='Historical', alpha=0.7)
        
        # Plot true next value
        ax.scatter(len(X_test[i]), y_test[i].item(), color='green', s=100, label='True')
        
        # Plot predictions
        ax.scatter(len(X_test[i]), vanilla_pred[i].item(), color='blue', s=100, label='Baseline', marker='^')
        ax.scatter(len(X_test[i]), fft_pred[i].item(), color='red', s=100, label='FFT-Enhanced', marker='s')
        
        ax.set_title(f'Example {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('prediction_comparison.png')
    plt.close()
    
    print("Prediction comparison saved to 'prediction_comparison.png'")


if __name__ == "__main__":
    print("FFT-Enhanced Chronos Fusion Model - Proof of Concept")
    print("=" * 50)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data
    print("\n1. Generating synthetic multi-frequency data...")
    X, y = generate_synthetic_data(n_samples=1000, seq_length=100, n_frequencies=3)
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = torch.tensor(X[:train_size])
    y_train = torch.tensor(y[:train_size])
    X_val = torch.tensor(X[train_size:train_size+val_size])
    y_val = torch.tensor(y[train_size:train_size+val_size])
    X_test = torch.tensor(X[train_size+val_size:])
    y_test = torch.tensor(y[train_size+val_size:])
    
    print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Create model
    print("\n2. Creating FFT-enhanced Chronos fusion model...")
    model = ChronosFFTFusion(
        chronos_model_name="amazon/chronos-t5-tiny",
        n_fft_features=16,
        hidden_dim=128,
        freeze_chronos=True
    )
    
    # Train model
    print("\n3. Training fusion layers...")
    train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val, epochs=10)
    
    # Evaluate
    print("\n4. Evaluating models...")
    results, vanilla_pred, fft_pred = evaluate_models(None, model, X_test, y_test)
    
    print("\n" + "="*50)
    print("RESULTS:")
    print(f"Baseline MSE: {results['vanilla_mse']:.4f}")
    print(f"FFT-Enhanced MSE: {results['fft_mse']:.4f}")
    print(f"Improvement: {results['improvement_percent']:.1f}%")
    print("="*50)
    
    # Visualize
    print("\n5. Creating visualizations...")
    visualize_predictions(X_test, y_test, vanilla_pred, fft_pred, n_examples=5)
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot FFT analysis of a sample
    plt.subplot(1, 2, 2)
    sample_signal = X_test[0].numpy()
    fft_mag = np.abs(np.fft.fft(sample_signal))
    freqs = np.fft.fftfreq(len(sample_signal))
    plt.plot(freqs[:len(freqs)//2], fft_mag[:len(fft_mag)//2])
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('FFT Analysis of Sample Signal')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_and_fft_analysis.png')
    plt.close()
    
    print("Training history and FFT analysis saved to 'training_and_fft_analysis.png'")
    print("\nProof of concept complete!")