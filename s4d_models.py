
import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from kernel_computations import fftconv_func, log_vandermonde_fast, make_bidirectional
from ssm_init import combination

_c2r = torch.view_as_real
_r2c = torch.view_as_complex

if tuple(map(int, torch.__version__.split(".")[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)


class OptimModule(nn.Module):
    """Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters"""

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class SSKernelDiag(OptimModule):
    """Version using (complex) diagonal state matrix (S4D)"""

    def __init__(
        self,
        H,
        N,
        n_ssm=None,
        channels=1,
        real_type="exp",
        lr=None,
        bandlimit=None,
        force_real=False,
        dt_min=0.001,
        dt_max=0.1,
    ):
        super().__init__()
        self.bandlimit = bandlimit
        self.real_type = real_type
        self.force_real = force_real

        dtype, cdtype = torch.float, torch.cfloat
        # Rank of low-rank correction
        self.H = H
        self.N = N
        self.n_ssm = n_ssm if n_ssm is not None else H
        assert self.H % self.n_ssm == 0
        self.repeat = self.H // self.n_ssm

        self.channels = channels

        A, _, B, _ = combination(
            measure="diag-lin",
            N=self.N,
            R=1,
            S=self.n_ssm,
        )

        C = torch.randn(channels, self.H, self.N // 2, dtype=cdtype)
        C = C * repeat(B, "t n -> (v t) n", v=self.H // self.n_ssm)
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))

        # Register parameters
        if lr is None or isinstance(lr, float):
            lr_dict = {}
        else:
            lr_dict, lr = lr, None

        log_dt = torch.rand(self.H, dtype=dtype) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        self.register("log_dt", log_dt, lr_dict.get("dt", lr))
        self.register("B", _c2r(B), lr_dict.get("B", lr))
        self.register("inv_A_real", self._A_init(A.real), lr_dict.get("A", lr))
        self.register("A_imag", A.imag, lr_dict.get("A", lr))

    def _A_init(self, A_real):
        A_real = torch.clamp(A_real, max=-1e-4)
        if self.real_type == "none":
            return -A_real
        elif self.real_type == "exp":
            return torch.log(-A_real)  # Some of the HiPPO methods have real part 0
        else:
            raise NotImplementedError

    def _A(self):
        # Get the internal A (diagonal) parameter
        if self.real_type == "none":
            A_real = -self.inv_A_real
        elif self.real_type == "exp":
            A_real = -torch.exp(self.inv_A_real)
        else:
            raise NotImplementedError
        A = A_real + 1j * self.A_imag
        return A

    def forward(self, L, bidirectional=False):
        """
        L: target length
        returns:
        (C, H, L) convolution kernel (generally C=1)
        """

        dt = torch.exp(self.log_dt)  # (H)
        C = _r2c(self.C)  # (C H N)
        A = self._A()  # (H N)

        B = _r2c(self.B)
        B = repeat(B, "t n -> 1 (v t) n", v=self.repeat)

        # Force A to be real valued, so the whole kernel can be interpreted as a "multi-head EMA"
        if self.force_real:
            A = A.real + 0j

        # Incorporate dt into A
        A = repeat(A, "t n -> (v t) n", v=self.repeat)
        A = rearrange(A, "H N -> 1 H N")
        dt = rearrange(dt, "H -> 1 H 1")
        dtA = A * dt  # (H N)

        C = (B * C).view(-1, self.H, self.N // 2)

        if self.channels > 1:
            C = rearrange(C, "c h n -> (c h) n")
            A = repeat(A, "1 h n -> 1 (c h) n", c=self.channels)
            dtA = repeat(dtA, "1 h n -> 1 (c h) n", c=self.channels)
        C = C * (torch.exp(dtA) - 1.0) / A

        # TODO (TD): make it work for C.shape[0] > 1
        if log_vandermonde_fast is not None and C.shape[0] == 1:
            K = log_vandermonde_fast(C.squeeze(0), dtA.squeeze(0), L)  # (H L)
        if self.channels > 1:
            K = K.view(self.channels, -1, L)

            if bidirectional:
                K = make_bidirectional(K, K.size(-1))  # (H L)
        return K


class S4D(nn.Module):
    def __init__(self, config, do_v_proj=False):
        super().__init__()

        self.d_model = config.d_model
        self.num_heads = config.num_ssm_heads
        self.bidirectional = config.bidirectional # not config.is_decoder
        self.gating = config.gating
        assert self.d_model % self.num_heads == 0
        self.H = self.d_model // self.num_heads
        self.N = config.d_state
        self.use_fast_fft_conv = config.use_fast_fft_conv
        self.do_v_proj = do_v_proj

        if self.use_fast_fft_conv:
            assert fftconv_func is not None, "Need to install fftconv."

        if self.gating:
            self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        if self.do_v_proj:
            self.v_proj = nn.Linear(self.d_model, self.d_model)

        if self.bidirectional:
            channels = 2
        else:
            channels = 1
        self.kernel = SSKernelDiag(
            self.H,
            N=self.N,
            channels=channels,
        )

        self.D = nn.Parameter(torch.randn(self.H))
        self.output_linear = nn.Linear(self.d_model, self.d_model)

    def fftconv_ref(self, k, ssm_kernel, q):
        seqlen = k.shape[-1]
        fft_size = 2 * seqlen
        k_f = torch.fft.rfft(k.to(dtype=ssm_kernel.dtype), n=fft_size) / fft_size
        ssm_kernel_f = torch.fft.rfft(ssm_kernel, n=fft_size)  # h L+1
        y = torch.fft.irfft(k_f * ssm_kernel_f, n=fft_size, norm="forward")[
            ..., :seqlen
        ]  # b d1 h l
        out = y + k * self.D.unsqueeze(-1)  # b h l

        return (out * q).to(dtype=k.dtype)

    def forward(self, u, attention_mask=None, output_attentions=False):
        # u (B L H)
        L_og = u.size(1)

        # use_fast_fft_conv needs L = 2 * k
        if self.use_fast_fft_conv and L_og % 2 != 0:
            u = F.pad(u, (0, 0, 0, 1))
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, 1))
        L = u.size(-2)

        # get the kernel for the convolution
        kernel = self.kernel(L=L, bidirectional=self.bidirectional)  # (H L)

        # .mT is the same as u = rearrange(u, "b h l -> b l h")
        k = self.k_proj(u).mT  # (B H L)
        if self.gating:
            q = self.q_proj(u).mT  # (B H L)
        else:
            q = torch.ones_like(k)
        if self.do_v_proj:
            v = self.v_proj(u).mT  # (B H L)
        else:
            v = torch.ones_like(q)

        if attention_mask is not None:
            # Make sure that padding is indeed zero (k_proj has a bias).
            k = k * rearrange(attention_mask, "b l -> b 1 l").to(dtype=k.dtype)
        if not self.use_fast_fft_conv:
            y = self.fftconv_ref(k=k, ssm_kernel=kernel, q=q)
        else:
            y = fftconv_func(
                u=k,
                kernel=kernel,
                v=v,
                q=q,
                D=self.D,
                num_heads=self.num_heads,
                gelu=False,
                force_fp16_output=torch.is_autocast_enabled(),
                output_hbl_layout=True,
            )

        if not torch.is_autocast_enabled():
            y = y.to(dtype=self.output_linear.weight.dtype)
        y = self.output_linear(y.mT)

        if L_og < L:
            y = y[:, :L_og, :]
        outputs = (y,)
        if output_attentions:
            outputs = outputs + (kernel,)

        return outputs
