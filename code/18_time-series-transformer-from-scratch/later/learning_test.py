test_cases = {
    "flat": torch.ones(1, 512) * 3.0,
    "line": torch.linspace(0, 5, 512).unsqueeze(0),
    "noisy_line": torch.linspace(0, 5, 512).unsqueeze(0) + torch.randn(1, 512) * 0.3,
    "sine": torch.sin(torch.linspace(0, 8 * math.pi, 512)).unsqueeze(0),
    "noisy_sine": torch.sin(torch.linspace(0, 8 * math.pi, 512)).unsqueeze(0)
    + torch.randn(1, 512) * 0.2,
}

# Inside training loop, every 50 steps:
if step % 50 == 0:
    model.eval()
    print(f"\n=== Step {step} | Loss: {loss:.4f} ===")
    for name, series in test_cases.items():
        mu, _ = model(series)
        # Grab the last patch prediction (what it thinks comes next)
        pred = mu[0, -1, :8].tolist()  # first 8 values of predicted patch
        actual = series[0, -32:].tolist()[:8]  # last 8 actual values for reference
        print(f"  {name:12s} | pred: {[f'{v:.2f}' for v in pred]}")
    model.train()


def grammar_test(model, test_cases, n_steps=32):
    """MSE per pattern — run periodically to see which patterns
    the model masters first."""
    model.eval()
    scores = {}
    for name, series in test_cases.items():
        # Use first 480 as context, last 32 as ground truth
        context = series[:, :480]
        truth = series[:, 480 : 480 + n_steps]
        result = model.forecast(context, n_steps=n_steps, n_samples=20)
        mse = ((result["median"] - truth) ** 2).mean().item()
        scores[name] = mse
    return scores


# Print table every 200 steps:
# Step  200 | flat: 0.001 | line: 0.450 | sine: 2.340
# Step  400 | flat: 0.001 | line: 0.032 | sine: 1.200
# Step  800 | flat: 0.001 | line: 0.008 | sine: 0.150
