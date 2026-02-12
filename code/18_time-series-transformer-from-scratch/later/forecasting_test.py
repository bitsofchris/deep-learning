# Dead simple test cases after training:

# Straight line
line = torch.linspace(0, 5, 512).unsqueeze(0)
result = model.forecast(line, n_steps=64)
# Should predict the line continuing upward

# Pure sine
t = torch.linspace(0, 8 * math.pi, 512).unsqueeze(0)
sine = torch.sin(t)
result = model.forecast(sine, n_steps=64)
# Should predict the wave continuing

# Sine + trend (harder)
rising_sine = torch.sin(t) + torch.linspace(0, 3, 512).unsqueeze(0)
result = model.forecast(rising_sine, n_steps=64)
