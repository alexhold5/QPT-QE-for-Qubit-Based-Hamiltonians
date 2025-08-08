import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, seq_len, min_dist, device):
        self.max_size = max_size
        self.device = device
        self.tokens = torch.empty((0, seq_len+1), device=device, dtype=int)
        self.energies = torch.empty((0, seq_len), device=device, dtype=torch.long)
        self.min_dist = min_dist

    def update(self, tokens, energies):
        combined_e = torch.cat((self.energies, energies), dim=0)
        combined_t = torch.cat((self.tokens, tokens), dim=0)

        # sorted_indices = torch.argsort(combined_e[:, -1])
        # combined_t, combined_e = combined_t[sorted_indices], combined_e[sorted_indices]

        # buffer1 = combined_t.unsqueeze(1)  # shape (N, 1, L)
        # buffer2 = combined_t.unsqueeze(0)  # shape (1, N, L)
        # hamming_matrix = (buffer1 != buffer2).sum(dim=-1)

        # selected = []
        # for i in range(combined_t.shape[0]):
        #     if not selected:
        #         selected.append(i)
        #     else:
        #         dists = hamming_matrix[i, selected]
        #         if torch.all(dists >= self.min_dist):
        #             selected.append(i)

        # Return pruned results
        # combined_t = combined_t[selected]
        # combined_e = combined_e[selected]

        combined_t_np = combined_t.view(combined_t.size(0), -1).cpu().numpy()
        unique_rows, unique_indices, counts = np.unique(
           combined_t_np, axis=0, return_index=True, return_counts=True
        )
        unique_indices = torch.tensor(unique_indices, device=combined_t.device)
        combined_t = torch.tensor(unique_rows, device=combined_t.device)
        combined_e = combined_e[unique_indices]

        if combined_e.shape[0] > self.max_size:
            best_idx = torch.topk(-combined_e[:, -1], k=self.max_size, dim=0).indices
            self.tokens, self.energies = combined_t[best_idx], combined_e[best_idx]
            # self.tokens = combined_t[:self.max_size]
            # self.energies = combined_e[:self.max_size]
        else:
            self.tokens, self.energies = combined_t, combined_e

    def sample(self, k=None, top_k=False):
        """
        Return tokens and energies as tensors
        """
        if self.tokens.shape[0] == 0 or k == 0:
            return None, None

        if k is None or k > len(self.tokens):
            k = len(self.tokens)

        idx = torch.randint(0, len(self.tokens), (k,), device=self.device)

        return self.tokens[idx], self.energies[idx]

