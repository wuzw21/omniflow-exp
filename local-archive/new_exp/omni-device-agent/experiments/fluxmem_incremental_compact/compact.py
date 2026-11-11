"""Storage-only adapter for original FluxMem decisions, fixed two-frame input blocks.

A ring holds the short window plus the incoming block. Once pruned, a frame
owns only selected features/absolute indices. Original per-frame merging may
modify those values once. No raw-feature audit copy is owned by this state.
"""
import torch
import torch.nn.functional as F
from state import storage_report

class CompactMemory:
    def __init__(self, policy, grid_hw=(8, 15)):
        self.policy = policy
        self.hw = grid_hw
        self.short_limit = max(1, int(policy.short_term_frames / 2))
        self.mid_limit = max(0, int(policy.mid_term_frames / 2))
        self.reset()

    def reset(self):
        self.count = 0
        self.short = []
        self.mid = []
        self.blocks = []
        self.norm = self.distance = self.keep = None
        self.height = self.width = self.offsets = None
        self.slot_indices = self.slot_grids = None
        self.indices = {}
        self.grids = {}

    def _allocate(self, feature):
        h, w = self.hw
        n = h*w
        slots = self.short_limit+1
        dev = feature.device
        self.norm = torch.empty((slots*n, feature.shape[1]), device=dev, dtype=feature.dtype)
        self.distance = torch.empty(slots*n, device=dev, dtype=feature.dtype)
        self.keep = torch.zeros(slots*n, device=dev, dtype=torch.bool)
        self.height = torch.arange(h, device=dev).repeat_interleave(w).repeat(slots)
        self.width = torch.arange(w, device=dev).repeat(h*slots)
        times = torch.arange(slots, device=dev).repeat_interleave(n)
        _, self.slot_indices, self.slot_grids = self.policy._frame_grids(
            torch.arange(slots*n, device=dev), times, self.height, self.width, self.hw)
        self.offsets = self.policy._neighbor_offsets(dev, include_center=True)

    def append(self, feature):
        h, w = self.hw
        n = h*w
        assert feature.shape[0] == n
        if self.norm is None:
            self._allocate(feature)
        i = self.count
        slot = i % (self.short_limit+1)
        cur = self.slot_indices[slot]
        # Slice copies touch only the incoming block, never the historical prefix.
        sl = slice(slot*n, (slot+1)*n)
        self.norm[sl].copy_(F.normalize(feature, p=2, dim=1, eps=1e-8))
        self.distance[sl].fill_(-1)
        self.keep[sl].zero_()
        self.indices[i] = cur
        self.grids[i] = self.slot_grids[slot]
        self.blocks.append({'indices': torch.arange(i*n, (i+1)*n, device=feature.device),
                            'features': feature.detach().clone()})
        self.short.append(i)
        if i > 0:
            paired, _, dist, _ = self.policy._max_sim_against_frames(
                query_indices=cur, neighbor_frames=[i-1], height_ids=self.height,
                width_ids=self.width, frame_grids=self.grids, hidden_norm=self.norm,
                offsets_3x3=self.offsets)
            if paired.numel() > 0:
                self.distance[paired] = dist
        if len(self.short) > self.short_limit:
            evicted = self.short.pop(0)
            self.policy._apply_adjacent_pruning(
                frame_id=evicted, buffer_frames=self.short, frame_to_indices=self.indices,
                frame_grids=self.grids, height_ids=self.height, width_ids=self.width,
                hidden_norm=self.norm, pair_distance_cache=self.distance,
                keep_mask=self.keep, pair_distance_threshold=None, offsets_3x3=self.offsets)
            selected = self.keep[self.indices[evicted]]
            block = self.blocks[evicted]
            self.blocks[evicted] = {k: v[selected] for k, v in block.items()}
            del self.indices[evicted], self.grids[evicted]
            self.mid.append(evicted)
            if len(self.mid) > self.mid_limit:
                self._merge(self.mid.pop(0))
        self.count += 1

    def _merge(self, frame):
        block = self.blocks[frame]
        values, absolute = block['features'], block['indices']
        n = len(absolute)
        dev = values.device
        spatial = absolute % (self.hw[0]*self.hw[1])
        keep = torch.ones(n, device=dev, dtype=torch.bool)
        self.policy._long_term_memory_merge_per_frames(
            batch_index=0, long_frames={4*frame+1}, visual_indices=torch.arange(n, device=dev),
            time_ids=torch.full((n,), 4*frame+1, device=dev, dtype=torch.long),
            height_ids=spatial // self.hw[1], width_ids=spatial % self.hw[1],
            hidden_states=values.unsqueeze(0), hidden_norm=None, keep_mask=keep, grid_hw=self.hw)
        self.blocks[frame] = {'indices': absolute[keep], 'features': values[keep]}

    def snapshot(self):
        # Packing selected output is paid once per query. Reading is side-effect free.
        return {k: torch.cat([b[k] for b in self.blocks]) for k in ['indices', 'features']}

    def state_report(self):
        payload = {k: v for k, v in vars(self).items() if k != 'policy'}
        return {**storage_report(payload), 'components': {
            k: storage_report(v)['tensor_storage_bytes'] for k, v in payload.items()}}
