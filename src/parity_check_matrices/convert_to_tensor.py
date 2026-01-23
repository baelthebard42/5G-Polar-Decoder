import torch

pcm = []

with open("PolarCode_N32_K11.txt", "r") as f:
    for line in f:
        # split, strip, and filter empty tokens
        numbers = line.strip().split()
        numbers_int = [int(x) for x in numbers]

        curr_row = torch.tensor(numbers_int, dtype=torch.long)
        pcm.append(curr_row)

pcm = torch.stack(pcm)
print(pcm.shape)
