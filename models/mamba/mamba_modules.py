import math

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


class Mamba(nn.Module):
    """ Full Mamba2 code from https://github.com/state-spaces/mamba """
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        d_head=64,
        d_ssm=None, # if not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        n_groups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rms_norm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0., float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None, # Absorb kwarg for general module
        process_group=None, # for model distribution
        sequence_parallel=True,
        device=None,
        dtype=None,
        learnable_init_states=False,
        activation="silu",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        # self.d_inner = self.expand * self.d_model
        self.d_head = d_head
        # self.n_groups = n_groups
        # self.n_heads = self.d_inner // self.d_head
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        self.n_groups = n_groups // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        assert self.n_groups % self.world_size == 0
        assert self.d_ssm % self.d_head == 0
        self.n_heads = self.d_ssm // self.d_head
        self.D_has_hdim = D_has_hdim
        self.rms_norm = rms_norm
        self.norm_before_gate = norm_before_gate
        
        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.n_groups * self.d_state + self.n_heads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)
        
        conv_dim = self.d_ssm + 2 * self.n_groups * self.d_state # x/ws, B, C
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv-1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True
        
        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.n_heads, self.d_head, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True
        
        self.act = nn.SiLU()
        
        # Initialize log dt bias
        dt = torch.exp(torch.rand(self.n_heads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True # just to be explicit
        
        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.n_heads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.n_heads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True
        
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.n_heads, device=device))
        self.D._no_weight_decay = True
        
        # Extra normalization layer right before output projection
        if self.rms_norm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        
        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)


    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.n_groups * self.d_state - self.n_heads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.n_groups * self.d_state, self.n_heads],
            dim=-1
        )

        # Conv step 
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.n_groups * self.d_state, self.n_groups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.n_groups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.d_head)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rms_norm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.d_head, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.d_head)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.d_head)
            D = repeat(self.D, "h -> h p", p=self.d_head)
            B = rearrange(B, "b (g n) -> b g n", g=self.n_groups)
            C = rearrange(C, "b (g n) -> b g n", g=self.n_groups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.d_head)
            if not self.rms_norm:
                z = rearrange(z, "b (h p) -> b h p", p=self.d_head)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rms_norm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rms_norm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.n_heads, self.d_head, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state


    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.n_heads,
                self.d_head,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
    
    
    def forward(self, u, seq_len=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        Args:
            u: (batch, seq_len, hidden_dim) if seq_len == None. If seq_len is not None, u is
               (batch*seq_len, hidden_dim). This is so that when we split u during sequence parallel,
               we split the batch*seq_len dimension (in case batch is small).
        Returns:
            same shape as u
        """
        seqlen_orig = seq_len
        if seq_len is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seq_len
        
        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out
            
        
        zxbcdt = self.in_proj(u) # (B, L, d_in_proj) or (B*L, d_in_proj)
        if seqlen_orig is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # NOTE: If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float()) # (n_heads) or (d_inner, d_state)
        initial_states = repeat(self.init_states, "... -> b ...", b=B) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0., float("inf")) else dict(dt_limit=self.dt_limit)
        
        if self.use_mem_eff_path and inference_params is None:
            # Fully fused path
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.d_head) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rms_norm else None,
                rmsnorm_eps=self.norm.eps if self.rms_norm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.d_head,
                ngroups=self.n_groups,
                norm_before_gate=self.norm_before_gate,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            if seqlen_orig is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            # z, xBC, dt = torch.split(
            #     zxbcdt, [self.d_inner, self.d_inner + 2 * self.n_groups * self.d_state, self.n_heads], dim=-1
            # )
            d_mlp = (zxbcdt.shape[-1] - (2 * self.d_ssm + 2 * self.n_groups * self.d_state + self.n_heads)) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt, [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.n_groups * self.d_state, self.n_heads], dim=-1
            )
            
            # dt = F.softplus(dt + self.dt_bias) # (B, L, n_heads)
            if conv_state is not None:
                if cu_seqlens is None:
                    # If we just take xBC[:, :, -self.d_conv:], it will cause error if seqlen < self.d_conv
                    # Instead, F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0))) # upadte state -> (B, D, W)
                else:
                    assert batch == 1, "varlen inference only supports batch dimension 1"
                    conv_varlen_states = causal_conv1d_varlen_states(xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1])
                    conv_state.copy_(conv_varlen_states)
            
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv-1)]
                ) # (B, L, self.d_ssm + 2 * n_groups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2), # (D, self.d_ssm + 2 * n_groups * d_state, L)
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2) # (D, L, self.d_ssm + 2 * n_groups * d_state)
            
            # Split into 3 main branches: X, B, C (respectively V, K, Q in the SSM/attention duality)
            x, B, C = torch.split(xBC, [self.d_ssm, self.n_groups*self.d_state, self.n_groups*self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.d_head),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.n_groups),
                rearrange(C, "b l (g n) -> b l g n", g=self.n_groups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.d_head) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.d_head) if not self.rms_norm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                initial_states=initial_states,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            )
            
            if ssm_state is not None:
                y, last_state, *rest = y
                if cu_seqlens is None:
                    ssm_state.copy_(last_state)
                else:
                    varlen_states = rest[0]
                    ssm_state.copy_(varlen_states)
            
            y = rearrange(y, "b l h p -> b l (h p)")
            
            if self.rms_norm:
                # Multiply "gate" branch and apply extra normalization layer
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_orig is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        return out


class MambaMine(nn.Module):
    """ My version of Mamba2 code from https://github.com/state-spaces/mamba """
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        d_head=64,
        d_ssm=None, # if not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        n_groups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rms_norm=True,
        norm_before_gate=False,
        dt_init_floor=1e-4,
        dt_limit=(0., float("inf")),
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None, # Absorb kwarg for general module
        device=None,
        dtype=None,
        learnable_init_states=False,
        activation="silu",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        # self.d_inner = self.expand * self.d_model
        self.d_head = d_head
        # self.n_groups = n_groups
        # self.n_heads = self.d_inner // self.d_head
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.d_inner = (self.expand * self.d_model)
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm
        self.n_groups = n_groups
        assert self.d_inner == self.expand * self.d_model
        assert self.d_ssm % self.d_head == 0
        self.n_heads = self.d_ssm // self.d_head
        self.D_has_hdim = D_has_hdim
        self.rms_norm = rms_norm
        self.norm_before_gate = norm_before_gate
        
        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.n_groups * self.d_state + self.n_heads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)
        
        conv_dim = self.d_ssm + 2 * self.n_groups * self.d_state # x, B, C
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=True,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv-1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True
        
        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.n_heads, self.d_head, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True
        
        # Initialize log dt bias
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(torch.rand(self.n_heads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True # just to be explicit (it's already True)
        
        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.n_heads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.n_heads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True
        
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.n_heads, device=device))
        self.D._no_weight_decay = True
        
        # Extra normalization layer right before output projection
        if self.rms_norm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)


    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.n_groups * self.d_state - self.n_heads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.n_groups * self.d_state, self.n_heads],
            dim=-1
        )

        # Conv step (NOTE: this code requires the causal_conv1d package)
        xBC = causal_conv1d_update(
            xBC,
            conv_state,
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.activation,
        )

        x, B, C = torch.split(xBC, [self.d_ssm, self.n_groups * self.d_state, self.n_groups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        A = repeat(A, "h -> h p n", p=self.d_head, n=self.d_state).to(dtype=torch.float32)
        dt = repeat(dt, "b h -> b h p", p=self.d_head)
        dt_bias = repeat(self.dt_bias, "h -> h p", p=self.d_head)
        D = repeat(self.D, "h -> h p", p=self.d_head)
        B = rearrange(B, "b (g n) -> b g n", g=self.n_groups)
        C = rearrange(C, "b (g n) -> b g n", g=self.n_groups)
        x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.d_head)
        if not self.rms_norm:
            z = rearrange(z, "b (h p) -> b h p", p=self.d_head)
        y = selective_state_update(
            ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rms_norm else None,
            dt_bias=dt_bias, dt_softplus=True
        )
        y = rearrange(y, "b h p -> b (h p)")
        if self.rms_norm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.n_heads, self.d_head, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state


    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.n_heads,
                self.d_head,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
    
    
    def forward(self, u, seq_len=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        Args:
            u: (batch, seq_len, hidden_dim) if seq_len == None. If seq_len is not None, u is
               (batch*seq_len, hidden_dim). This is so that when we split u during sequence parallel,
               we split the batch*seq_len dimension (in case batch is small).
            seq_len, seq_idx, cu_seqlens: arguments for processing variable length input.
        Returns:
            same shape as u
        """
        seqlen_orig = seq_len
        if seq_len is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seq_len
        
        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out
        
        zxbcdt = self.in_proj(u) # (B, L, d_in_proj) or (B*L, d_in_proj)
        if seqlen_orig is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # NOTE: If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float()) # (n_heads) or (d_inner, d_state)
        initial_states = repeat(self.init_states, "... -> b ...", b=B) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0., float("inf")) else dict(dt_limit=self.dt_limit)
        
        if self.use_mem_eff_path and inference_params is None:
            # Fully fused path
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.d_head) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rms_norm else None,
                rmsnorm_eps=self.norm.eps if self.rms_norm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.d_head,
                ngroups=self.n_groups,
                norm_before_gate=self.norm_before_gate,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            if seqlen_orig is not None:
                out = rearrange(out, "b l d -> (b l) d")
        else:
            # z, xBC, dt = torch.split(
            #     zxbcdt, [self.d_inner, self.d_inner + 2 * self.n_groups * self.d_state, self.n_heads], dim=-1
            # )
            d_mlp = (zxbcdt.shape[-1] - (2 * self.d_ssm + 2 * self.n_groups * self.d_state + self.n_heads)) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt, [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.n_groups * self.d_state, self.n_heads], dim=-1
            )
            
            # dt = F.softplus(dt + self.dt_bias) # (B, L, n_heads)
            if conv_state is not None:
                if cu_seqlens is None:
                    # If we just take xBC[:, :, -self.d_conv:], it will cause error if seqlen < self.d_conv
                    # Instead, F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0))) # upadte state -> (B, D, W)
                else:
                    assert batch == 1, "varlen inference only supports batch dimension 1"
                    conv_varlen_states = causal_conv1d_varlen_states(xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1])
                    conv_state.copy_(conv_varlen_states)
            
            assert self.activation in ["silu", "swish"]
            # NOTE: this code requires the causal_conv1d package
            xBC = causal_conv1d_fn(
                x=xBC.transpose(1, 2), # (D, self.d_ssm + 2 * n_groups * d_state, L)
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=seq_idx,
            ).transpose(1, 2) # (D, L, self.d_ssm + 2 * n_groups * d_state)
            
            # Split into 3 main branches: X, B, C (respectively V, K, Q in the SSM/attention duality)
            x, B, C = torch.split(xBC, [self.d_ssm, self.n_groups*self.d_state, self.n_groups*self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.d_head),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.n_groups),
                rearrange(C, "b l (g n) -> b l g n", g=self.n_groups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.d_head) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.d_head) if not self.rms_norm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                initial_states=initial_states,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            )
            
            if ssm_state is not None:
                y, last_state, *rest = y
                if cu_seqlens is None:
                    ssm_state.copy_(last_state)
                else:
                    varlen_states = rest[0]
                    ssm_state.copy_(varlen_states)
            
            y = rearrange(y, "b l h p -> b l (h p)")
            
            if self.rms_norm:
                # Multiply "gate" branch and apply extra normalization layer
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_orig is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        return out


class MambaMineSimple(nn.Module):
    """ My version of Mamba2 code from https://github.com/state-spaces/mamba.
    No device & dtype, conv_init, D_has_hdim, learnable_init_states, and inference_params.
    """
    def __init__(
        self,
        dim_model: int,
        dim_state: int = 128,
        dim_conv: int = 4,
        expand: int = 2,
        dim_head: int = 64,
        dim_ssd: int = None, # if not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        num_group: int = 1,
        A_init_range: tuple = (1, 16),
        dt_init_floor: float = 1.0e-4,
        dt_limit: tuple = (0., float("inf")),
        chunk_size: int = 256,
        use_mem_eff_path: bool = True,
        layer_idx = None, # absorb kwarg for general module
        rms_norm=True,
        activation: str = "swish",
    ) -> None:
        super().__init__()
        
        self.dim_model = dim_model
        self.dim_state = dim_state
        self.dim_conv = dim_conv
        self.dim_inner = expand * self.dim_model
        self.dim_head = dim_head
        self.dim_ssd = self.dim_inner if dim_ssd is None else dim_ssd
        self.num_group = num_group
        assert self.dim_inner == expand * self.dim_model
        assert self.dim_ssd % self.dim_head == 0
        self.num_head = self.dim_ssd // self.dim_head # num_head is 8 for original model
        self.dt_limit = dt_limit
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.rms_norm = rms_norm
        self.activation = activation
        
        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.dim_inner + 2 * self.num_group * self.dim_state + self.num_head
        self.in_proj = nn.Linear(self.dim_model, d_in_proj, bias=False)
        
        conv_dim = self.dim_ssd + 2 * self.num_group * self.dim_state # x, B, C
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, dim_conv, bias=True, groups=conv_dim, padding=dim_conv-1)
        
        # initialize log dt bias
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(torch.rand(self.num_head) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True # just to be explicit (it's already True)
        
        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.num_head).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.num_head, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True
        
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.num_head))
        self.D._no_weight_decay = True

        # extra normalization layer right before output projection
        if self.rms_norm:
            self.norm = RMSNormGated(self.dim_ssd, eps=1.0e-5, norm_before_gate=False)
        
        self.out_proj = nn.Linear(self.dim_inner, self.dim_model, bias=False)
    
    
    def forward(self, u, seq_len=None, seq_idx=None, cu_seqlens=None):
        """
        Args:
            u: (batch, seq_len, hidden_dim) if seq_len == None. If seq_len is not None, u is
               (batch*seq_len, hidden_dim). This is so that when we split u during sequence parallel,
               we split the batch*seq_len dimension (in case batch is small).
            memory: (batch, mem_seq_len, mem_hidden_dim) tokens for cross-attention
            t: (batch, t_hidden_dim) tokens for FiLM containing timestep embedding.
            seq_len, seq_idx, cu_seqlens: arguments for processing variable length input.
        Returns:
            same shape as u
        """
        # TODO: maybe assign specific dtype for some parameters
        seqlen_orig = seq_len
        if seq_len is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seq_len

        zxbcdt = self.in_proj(u) # (B, L, d_in_proj) or (B*L, d_in_proj)
        if seqlen_orig is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # NOTE: If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float()) # (num_head) or (dim_inner, dim_state)
        dt_limit_kwargs = {} if self.dt_limit == (0., float("inf")) else dict(dt_limit=self.dt_limit)
        
        if self.use_mem_eff_path:
            # Fully fused path
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rms_norm else None,
                rmsnorm_eps=self.norm.eps if self.rms_norm else 1.0e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.dim_head,
                ngroups=self.num_group,
                norm_before_gate=False,
                **dt_limit_kwargs,
            )
            if seqlen_orig is not None:
                out = rearrange(out, "b l d -> (b l) d")
        else:
            dim_mlp = (zxbcdt.shape[-1] - (2*self.dim_ssd + 2*self.num_group*self.dim_state + self.num_head)) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt, [dim_mlp, dim_mlp, self.dim_ssd, self.dim_ssd + 2*self.num_group*self.dim_state, self.num_head], dim=-1
            )
            
            assert self.activaiton in ["silu", "swish"]
            # NOTE: this code requires the causal_conv1d package
            xBC = causal_conv1d_fn(
                x=xBC.transpose(1, 2), # (D, self.dim_ssd + 2 * n_groups * d_state, L)
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=seq_idx,
            ).transpose(1, 2) # (D, L, self.dim_ssd + 2 * n_groups * d_state)
            
            # split into 3 main branches: X, B, C (respectively V, K, Q in the SSM/attention duality)
            x, B, C = torch.split(xBC, [self.dim_ssd, self.num_group*self.dim_state, self.num_group*self.dim_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.dim_head),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.num_group),
                rearrange(C, "b l (g n) -> b l g n", g=self.num_group),
                chunk_size=self.chunk_size,
                D=self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.dim_head) if not self.rms_norm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
            )
            y = rearrange(y, "b l h p -> b l (h p)")
            
            if self.rms_norm:
                y = self.norm(y, z) # multiply "gate" branch and apply extra normalization layer
            if dim_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_orig is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        return out


class Mamba2Simple(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=2,
        d_head=128,
        n_groups=1,
        A_init_range=(1, 16),
        dt_init_floor=1.0e-4,
        dt_limit=(0., float("inf")),
        learnable_init_states=False,
        activation="swish",
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None, # absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.d_head = d_head
        self.n_groups = n_groups
        assert self.d_inner % self.d_head == 0
        self.n_heads = self.d_inner // self.d_head
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        
        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.n_groups * self.d_state + self.n_heads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)
        
        conv_dim = self.d_inner + 2 * self.n_groups * self.d_state
        self.conv1d = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim, kernel_size=d_conv, bias=True,
                                groups=conv_dim, padding=d_conv-1, **factory_kwargs)
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True
        
        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.n_heads, self.d_head, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True
        
        # initialize log dt bias
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(torch.rand(self.n_heads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True # just to be explicit
        
        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.n_heads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True
        
        # extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_ssm, eps=1.0e-5, norm_before_gate=False, **factory_kwargs)
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)
    
    
    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch_size, seq_len, dim = u.shape
        
        zxbcdt = self.in_proj(u) # (B, L, d_in_proj)
        A = -torch.exp(self.A_log) # (n_heads) or (d_inner, d_state)
        initial_states = repeat(self.init_states, "... -> b ...", b=batch_size) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0., float("inf")) else dict(dt_limit=self.dt_limit)
        
        if self.use_mem_eff_path:
            # Fully fused path
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.d_head,
                ngroups=self.n_groups,
                norm_before_gate=False,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
        else:
            z, xBC, dt = torch.split(zxbcdt, [self.d_inner, self.d_inner+2*self.n_groups*self.d_state, self.n_heads], dim=-1)
            dt = F.softplus(dt + self.dt_bias) # (B, L, n_heads)
            assert self.activation in ["silu", "swish"]
            
            # 1d conv: Note that this code supposes using causal_conv1d library
            xBC = causal_conv1d_fn(
                x=xBC.transpose(1, 2), # (B, d_inner+2*n_groups*d_state, L)
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation
            )
            
            # Split into 3 main branches: X, B, C (respectively V, K, Q in the SSM/attention duality)
            x, B, C = torch.split(xBC, [self.d_inner, self.n_groups*self.d_state, self.n_groups*self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.d_head),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.n_groups),
                rearrange(C, "b l (g n) -> b l g n", g=self.n_groups),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            y = rearrange(y, "b l h p -> b l (h p)")
            
            # multiply "gate" brnach and apply extra normalization layer
            y = self.norm(y, z)
            out = self.out_proj(y)
        return out