import math

import torch
import torch.nn.functional as F
from cauchy_mult import vand_log_mult_sym_bwd, vand_log_mult_sym_fwd
from einops import rearrange
from fftconv import fftconv_bwd, fftconv_fwd


class LogVandMultiplySymmetric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, x, L):
        batch, N = v.shape
        supported_N_values = [1 << log_n for log_n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        if not N in supported_N_values:
            raise NotImplementedError(f"Only support N values in {supported_N_values}")
        max_L_value = 32 * 1024 * 64 * 1024
        if L > max_L_value:
            raise NotImplementedError(f"Only support L values <= {max_L_value}")
        if not v.is_cuda and x.is_cuda:
            raise NotImplementedError(f"Only support CUDA tensors")
        ctx.save_for_backward(v, x)
        return vand_log_mult_sym_fwd(v, x, L)

    @staticmethod
    def backward(ctx, dout):
        v, x = ctx.saved_tensors
        dv, dx = vand_log_mult_sym_bwd(v, x, dout)
        return dv, dx, None


if (vand_log_mult_sym_fwd and vand_log_mult_sym_bwd) is not None:
    log_vandermonde_fast = LogVandMultiplySymmetric.apply
else:
    log_vandermonde_fast = None


def make_bidirectional(ssm_kernel, L):
    k_direct, k_reversed = torch.tensor_split(ssm_kernel, 2, dim=0)

    return (
        F.pad(k_direct.contiguous(), (0, L)).contiguous()
        + torch.roll(
            F.pad(k_reversed.flip(-1).contiguous(), (L, 0)), 1, dims=-1
        ).contiguous()
    ).squeeze(0)


class FFTConvFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u,
        k,
        D,
        dropout_mask=None,
        gelu=True,
        force_fp16_output=False,
        output_hbl_layout=False,
        v=None,
        head_dim=1,
        q=None,
        fftfp16=False,
        k_rev=None,
    ):
        seqlen = u.shape[-1]
        fft_size = max(2 * 2 ** int(math.ceil(math.log2(seqlen))), 16)
        k_f = torch.fft.rfft(k, n=fft_size)
        if k_rev is not None:
            k_f = k_f + torch.fft.rfft(k_rev, n=fft_size).conj()
        if u.stride(-1) != 1:
            u = u.contiguous()
        k_f = k_f.contiguous()
        D = D.contiguous()
        if v is not None and v.stride(-1) != 1:
            v = v.contiguous()
        if q is not None and q.stride(-1) != 1:
            q = q.contiguous()
        if dropout_mask is not None:
            dropout_mask = dropout_mask.contiguous()
        ctx.save_for_backward(u, k_f, D, dropout_mask, v, q)
        ctx.output_hbl_layout = output_hbl_layout
        ctx.head_dim = head_dim
        ctx.gelu = gelu
        ctx.fftfp16 = fftfp16
        ctx.has_k_rev = k_rev is not None
        ctx.klen = k.shape[-1]
        out = fftconv_fwd(
            u,
            k_f,
            D,
            v,
            head_dim,
            q,
            dropout_mask,
            gelu,
            False,
            False,
            fft_size,
            force_fp16_output,
            output_hbl_layout,
            fftfp16,
        )
        return out

    @staticmethod
    def backward(ctx, dout):
        if ctx.output_hbl_layout:
            dout = rearrange(
                rearrange(dout, "b h l -> h b l").contiguous(), "h b l -> b h l"
            )
        else:
            dout = dout.contiguous()
        u, k_f, D, dropout_mask, v, q = ctx.saved_tensors
        seqlen = u.shape[-1]
        fft_size = max(2 * 2 ** int(math.ceil(math.log2(seqlen))), 16)
        du, dk_f, dD, dv, dq = fftconv_bwd(
            dout,
            u,
            k_f,
            D,
            v,
            ctx.head_dim,
            q,
            dropout_mask,
            ctx.gelu,
            False,
            False,
            fft_size,
            ctx.output_hbl_layout,
            ctx.fftfp16,
        )
        klen = ctx.klen
        dk = torch.fft.irfft(dk_f, n=fft_size, norm="forward")[..., :klen]
        dk_rev = (
            None
            if not ctx.has_k_rev
            else torch.fft.irfft(dk_f.conj(), n=fft_size, norm="forward")[..., :seqlen]
        )
        if v is not None:
            dv = dv.to(
                dtype=v.dtype
            )  # We do atomicAdd in fp32 so might need to convert to fp16
        return (
            du,
            dk,
            dD,
            None,
            None,
            None,
            None,
            dv if v is not None else None,
            None,
            dq if q is not None else None,
            None,
            dk_rev,
        )


def fftconv_func(
    u,
    kernel,
    v,
    q,
    D,
    gelu=True,
    force_fp16_output=False,
    output_hbl_layout=False,
    num_heads=1,
    fftfp16=False,
):
    return FFTConvFunc.apply(
        u,
        kernel,
        D,
        None,
        gelu,
        force_fp16_output,
        output_hbl_layout,
        v,
        num_heads,
        q,
        fftfp16,
        None,
    )
