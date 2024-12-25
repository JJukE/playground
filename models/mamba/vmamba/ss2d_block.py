import math

import torch
from torch import nn

from .vss_block_from_vmamba import Permute

class SS2DBlock:
    def __init__(
        self,
        d_model=96,
        d_state=16, # in mamba2, dstate should be bigger
        ssm_ratio=2.,
        dt_rank="auto",
        d_conv=3, # < 2 means no conv
        conv_bias=True,
        dropout=0.,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.,
        dt_init_floor=1e-4,
        with_initial_state=False,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        assert d_inner % dt_rank == 0
        
        self.with_dconv = d_conv > 1
        
        # in proj
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = nn.Linear(d_model, d_proj, bias=bias)
        self.act = nn.GeLU()
        
        # conv
        if self.with_dconv:
            self.conv2d = nn.Sequential(
                Permute(0, 3, 1, 2),
                nn.Conv2d(in_channels=d_inner, out_channels=d_inner, kernel_size=d_conv,
                          groups=d_inner, bias=conv_bias, padding=(d_conv - 1) // 2, **factory_kwargs),
                Permute(0, 2, 3, 1)
            )
        
        # x_proj
        k_group = 4
        self.x_proj = [nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False) for _ in range(k_group)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj
        self.out_act = nn.GELU() if kwargs["oact"] == True else nn.Identity() # TODO
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
        # initialize
        self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
        # self.A_logs = nn.Parameter(torch.randn((k_group, dt_rank))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1 # v1
        self.A_logs = nn.Parameter(torch.zeros((k_group, dt_rank))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, dt_rank)))
        
        self.initial_state = None
        if with_initial_state:
            self.inital_state = nn.Parameter(torch.zeros((1, k_group * dt_rank, int(d_inner // dt_rank), d_state)), requires_grad=False)
    
    
    def forward_core(
        self,
        x: torch.Tensor = None,
        force_fp32 = False, # True: input fp32
        chunk_size = 64,
        dstate = 64,
        selective_scan_backend = None,
        scan_mode = "cross2d",
        scan_force_torch = False,
        **kwargs,
    ):
        assert scan_mode in ["unidi", "bidi", "cross2d"]
        assert selective_scan_backend in [None, "triton", "torch"]
        x_proj_bias = getattr(self, "x_proj_bias", None)
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        
        N = dstate
        B, H, W, RD = x.shape
        K, R = self.A_logs.shape
        K, R, D = self.Ds.shape
        assert RD == R * D
        L = H * W
        KR = K * R
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]
        
        initial_state = None
        if self.initial_state is not None:
            assert self.initial_state.shape[-1] == dstate
            initial_state = self.initial_state.detach().repeat(B, 1, 1, 1)
        xs = cross_scan_fn(x.view(B, H, W, RD), in_channel_first=False, out_channel_first=False,
                           scans=_scan_mode, force_torch=scan_force_torch) # (B, H, W, 4, D)