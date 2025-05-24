## Run 33

# Weighted time-step loss (more focus on early steps)
weights = torch.tensor([5, 4, 3, 2, 1, 1, 1, 1, 1], device=pred.device, dtype=pred.dtype)
weights = weights / weights.sum()  # Normalize

pred_seq = idx_pred[:, -tsGPT_obj.block_size:, :]  # [B, 9, F]
target_seq = yb                                     # [B, 9, F]

loss_all = ((pred_seq - target_seq) ** 2 * weights.view(1, -1, 1)).mean()

