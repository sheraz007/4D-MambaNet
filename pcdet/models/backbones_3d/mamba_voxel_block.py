import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

from pcdet.models.model_utils.pointcept_utils.serialization import encode as serialization_encode


def _attention_dims(d_model):
    """d_inner / num_heads so attention (+pos) params stay within ±5% of Mamba."""
    if d_model == 64:
        return 128, 8
    if d_model == 128:
        return 224, 8
    # Generic fallback: match Mamba expand=2 width; heads divide d_inner.
    d_inner = 2 * d_model
    for h in (8, 16, 4, 32):
        if d_inner % h == 0:
            return d_inner, h
    return d_inner, 1


class MambaVoxelBlock(nn.Module):
    """Sparse-voxel Mamba (or attention) residual block for 4D radar backbones.

    Applies ``F' = F + Mixer(LN(F))`` on a ``spconv.SparseConvTensor`` feature
    matrix ``(N, C)``. The paper default uses Mixer = Mamba SSM over all active
    voxels of the mini-batch as one sequence
    (``PER_FRAME_SEQUENCE=False``, ``SCAN_ORDER='native'``, ``MIXER='mamba'``).

    Ablation knobs (configs under ``tools/cfgs/vod/ablation_*``):
      - ``per_frame_sequence`` / ``scan_order`` / ``bidirectional``: sequence layout
      - ``mixer='attention'``: global MHSA per frame (parameter-matched width)
      - ``use_fg_scoring``: score-sort voxels before Mamba (research option)
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 use_fg_scoring=False, per_frame_sequence=False,
                 scan_order='native', scan_seed=0, bidirectional=False,
                 mixer='mamba'):
        super().__init__()
        self.use_fg_scoring = use_fg_scoring
        self.per_frame_sequence = bool(per_frame_sequence)
        self.scan_order = str(scan_order).lower()
        self.scan_seed = int(scan_seed)
        self.bidirectional = bool(bidirectional)
        self.mixer = str(mixer).lower()
        if self.scan_order not in ('native', 'shuffle', 'hilbert', 'zorder'):
            raise ValueError(f'unsupported SCAN_ORDER: {scan_order}')
        if self.mixer not in ('mamba', 'attention'):
            raise ValueError(f'unsupported MIXER: {mixer}')
        if self.mixer == 'attention' and use_fg_scoring:
            raise ValueError('use_fg_scoring is not supported with MIXER=attention')
        if self.mixer == 'attention' and self.bidirectional:
            raise ValueError('BIDIRECTIONAL is not supported with MIXER=attention')

        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

        if self.mixer == 'attention':
            d_inner, num_heads = _attention_dims(d_model)
            if d_inner % num_heads != 0:
                raise ValueError(f'd_inner={d_inner} not divisible by num_heads={num_heads}')
            self.d_inner = d_inner
            self.num_heads = num_heads
            self.q_proj = nn.Linear(d_model, d_inner)
            self.k_proj = nn.Linear(d_model, d_inner)
            self.v_proj = nn.Linear(d_model, d_inner)
            self.out_proj = nn.Linear(d_inner, d_model)
            self.coord_enc = nn.Linear(3, d_model)
        else:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            if self.bidirectional:
                self.mamba_bwd = Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            if use_fg_scoring:
                self.fg_scorer = nn.Linear(d_model, 1)

    def _run_mamba(self, seq):
        """seq (1, L, C) -> (1, L, C). Backward Mamba is added before the residual."""
        out = self.mamba(seq)
        if self.bidirectional:
            out = out + self.mamba_bwd(seq.flip(1)).flip(1)
        return out

    def _inverse_perm(self, perm):
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.shape[0], device=perm.device)
        return inv

    def _frame_perm(self, frame_indices, batch_id, depth):
        """Permutation of one frame. Inverse restores the original row order."""
        n = frame_indices.shape[0]
        device = frame_indices.device
        if self.scan_order == 'native':
            return torch.arange(n, device=device)
        if self.scan_order == 'shuffle':
            generator = torch.Generator(device='cpu')
            generator.manual_seed(self.scan_seed + int(batch_id))
            return torch.randperm(n, generator=generator).to(device)
        zyx = frame_indices[:, 1:4].long()
        order_name = 'hilbert' if self.scan_order == 'hilbert' else 'z'
        code = serialization_encode(zyx, batch=None, depth=depth, order=order_name)
        return torch.argsort(code)

    def _sequence_mamba(self, indices, feats, depth):
        # Non-native scan orders are always one sequence per frame.
        per_frame = self.per_frame_sequence or self.scan_order != 'native'
        if not per_frame:
            return self._run_mamba(feats.unsqueeze(0)).squeeze(0)

        out = feats.new_empty(feats.shape)
        for batch_id in indices[:, 0].unique():
            rows = torch.nonzero(indices[:, 0] == batch_id, as_tuple=False).squeeze(1)
            perm = self._frame_perm(indices.index_select(0, rows), batch_id, depth)
            seq = feats.index_select(0, rows).index_select(0, perm).unsqueeze(0)
            y = self._run_mamba(seq).squeeze(0)
            y = y.index_select(0, self._inverse_perm(perm))
            out.index_copy_(0, rows, y)
        return out

    def _attention_frame(self, h, coords_zyx, spatial_shape):
        """Global MHSA over one frame. h: (L, C) already LayerNorm'd."""
        L = h.shape[0]
        if L == 0:
            return h
        spatial = h.new_tensor(spatial_shape, dtype=h.dtype)
        pos = self.coord_enc(coords_zyx.to(dtype=h.dtype) / spatial)
        h_pos = h + pos
        q = self.q_proj(h_pos)
        k = self.k_proj(h_pos)
        v = self.v_proj(h_pos)
        head_dim = self.d_inner // self.num_heads
        # (1, heads, L, head_dim) for scaled_dot_product_attention
        q = q.view(1, L, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(1, L, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(1, L, self.num_heads, head_dim).transpose(1, 2)
        # Memory-efficient SDPA path in fp32
        q32, k32, v32 = q.float(), k.float(), v.float()
        y = F.scaled_dot_product_attention(q32, k32, v32)
        y = y.to(dtype=h.dtype)
        y = y.transpose(1, 2).reshape(L, self.d_inner)
        return self.out_proj(y)

    def _sequence_attention(self, indices, feats, spatial_shape):
        """Always one global attention sequence per frame (never across frames)."""
        out = feats.new_empty(feats.shape)
        for batch_id in indices[:, 0].unique():
            rows = torch.nonzero(indices[:, 0] == batch_id, as_tuple=False).squeeze(1)
            h = feats.index_select(0, rows)
            coords = indices.index_select(0, rows)[:, 1:4]
            y = self._attention_frame(h, coords, spatial_shape)
            out.index_copy_(0, rows, y)
        return out

    def forward(self, x):
        feats = x.features          # (N, C)
        residual = feats

        feats = self.norm(feats)

        if self.mixer == 'attention':
            feats = self._sequence_attention(x.indices, feats, x.spatial_shape)
            feats = feats + residual
        elif self.use_fg_scoring:
            scores = self.fg_scorer(feats).sigmoid()      # (N, 1)
            sort_idx = torch.argsort(scores.squeeze(), descending=True)
            feats_sorted = feats[sort_idx]
            feats_sorted = self.mamba(feats_sorted.unsqueeze(0)).squeeze(0)
            unsort_idx = torch.argsort(sort_idx)
            feats = feats_sorted[unsort_idx]
            feats = feats * scores + residual * (1 - scores)
        elif (not self.per_frame_sequence) and self.scan_order == 'native' and (not self.bidirectional):
            feats = self.mamba(feats.unsqueeze(0)).squeeze(0)
            feats = feats + residual
        else:
            depth = min(max(int(max(x.spatial_shape)).bit_length(), 1), 16)
            feats = self._sequence_mamba(x.indices, feats, depth)
            feats = feats + residual

        return x.replace_feature(feats)
