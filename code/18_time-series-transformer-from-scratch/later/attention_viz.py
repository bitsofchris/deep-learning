# In CausalSelfAttention.forward, after softmax:
self.last_attn = attn.detach()  # save it

# Then after training:
model.eval()
model(test_cases["sine"])
attn = model.blocks[-1].attn.last_attn[0, 0]  # first head, last layer
print("Attention pattern (last layer, head 0):")
for i in range(attn.size(0)):
    row = attn[i].tolist()
    print(f"  patch {i:2d} attends to: {[f'{v:.2f}' for v in row]}")
