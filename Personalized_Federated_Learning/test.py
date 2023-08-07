import torch

'''
N, C, H, W = 2, 3, 5, 5
x = torch.randn(N, C, H, W)

tmp = x.view(N, C, -1)
min_vals = tmp.min(2, keepdim=True).values
tmp = (tmp - min_vals) / min_vals #max_vals
max_vals = tmp.max(2, keepdim=True).values
tmp = tmp / max_vals

x = tmp.view(x.size())

for n in range(N):
    for c in range(C):
        x_ = x[n, c]
        print(n, c, x_.shape, x_.min(), x_.max())
'''

'''
tmp = torch.randn(5, 3)
min_vals = tmp.min(2, keepdim=True).values
tmp = (tmp - min_vals) / min_vals #max_vals
max_vals = tmp.max(2, keepdim=True).values
tmp = tmp / max_vals

print(tmp)
print(torch.linalg.norm(tmp))
'''


s_temp = tmp = torch.randn(5, 3) #torch.tensor([[5,6,7],[7,6,5],[8,8,9]])
print("s_temp")
print(s_temp)
print()
# Compute min and max values
min_values_rows, _ = torch.min(s_temp, dim=0)
# Get the minimum value along columns (axis 1)
min_values_cols, _ = torch.min(s_temp, dim=1)
print(f"min_values_rows: {min_values_rows}")
print(f"min_values_cols: {min_values_cols}")
min_val = torch.min(torch.min(min_values_rows), torch.min(min_values_cols))
print(min_val)
print(f"min_val: {min_val}")

max_values_rows, _ = torch.max(s_temp, dim=0)
# Get the minimum value along columns (axis 1)
max_values_cols, _ = torch.max(s_temp, dim=1)
print(f"max_values_rows: {max_values_rows}")
print(f"max_values_cols: {max_values_cols}")
max_val = torch.max(torch.max(max_values_rows), torch.max(max_values_cols))
print(f"max_val: {max_val}")

print()

# Min-Max Scaling
print("s_inter")
s_inter = (s_temp - min_val)
print(s_inter)
print()
print("s_normed")
s_normed = s_inter / (max_val - min_val)
print(s_normed)
print()
print(f"s norm: {torch.linalg.norm(s_normed)}")
print()
print()
print()
print("Direct norm eqn")
print((s_temp - min_val) / (max_val - min_val))


# This is still returning norms of up to 2.25... why doesn't this work? Too few nums?