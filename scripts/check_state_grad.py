"""FORK check: tece li grad kroz chunk_gdn2 initial_state?

Pokreni na GPU (lokalno ili Modal):
  modal run modal_app.py::check_state_grad
"""

import torch

RESULT = {"grad_flows": None, "error": None}


def _run_check():
    try:
        from fla.ops.gdn2 import chunk_gdn2

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        bsz, seq, heads, dim = 2, 64, 4, 32
        requires_grad = device.type == "cuda"

        q1 = torch.randn(bsz, seq, heads, dim, device=device, dtype=dtype, requires_grad=requires_grad)
        k1 = torch.randn(bsz, seq, heads, dim, device=device, dtype=dtype, requires_grad=requires_grad)
        v1 = torch.randn(bsz, seq, heads, dim, device=device, dtype=dtype, requires_grad=requires_grad)
        g1 = -torch.rand(bsz, seq, heads, dim, device=device, dtype=torch.float32)
        b1 = torch.rand(bsz, seq, heads, dim, device=device, dtype=dtype, requires_grad=requires_grad)
        w1 = torch.rand(bsz, seq, heads, dim, device=device, dtype=dtype, requires_grad=requires_grad)

        delta1, state1 = chunk_gdn2(
            q=q1,
            k=k1,
            v=v1,
            g=g1,
            b=b1,
            w=w1,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
        )

        q2 = torch.randn(bsz, seq, heads, dim, device=device, dtype=dtype, requires_grad=requires_grad)
        k2 = torch.randn(bsz, seq, heads, dim, device=device, dtype=dtype, requires_grad=requires_grad)
        v2 = torch.randn(bsz, seq, heads, dim, device=device, dtype=dtype, requires_grad=requires_grad)
        g2 = -torch.rand(bsz, seq, heads, dim, device=device, dtype=torch.float32)
        b2 = torch.rand(bsz, seq, heads, dim, device=device, dtype=dtype, requires_grad=requires_grad)
        w2 = torch.rand(bsz, seq, heads, dim, device=device, dtype=dtype, requires_grad=requires_grad)

        delta2, _ = chunk_gdn2(
            q=q2,
            k=k2,
            v=v2,
            g=g2,
            b=b2,
            w=w2,
            initial_state=state1,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
        )

        loss = delta2.sum()
        loss.backward()

        grad_norms = {
            "q1": q1.grad.norm().item() if q1.grad is not None else None,
            "q2": q2.grad.norm().item() if q2.grad is not None else None,
        }

        if device.type == "cuda" and grad_norms["q1"] is not None and grad_norms["q1"] > 0:
            RESULT["grad_flows"] = True
            print(f"PASS: grad tece kroz initial_state (q1.grad = {grad_norms['q1']:.6f})")
        elif device.type == "cpu":
            RESULT["grad_flows"] = "CPU_SKIP"
            print("SKIP: CPU run — nema GPU kernela; grad check nije validan bez CUDA.")
        else:
            RESULT["grad_flows"] = False
            print(f"FAIL: grad NE tece kroz initial_state (q1.grad = {grad_norms['q1']})")

        print(
            f"device={device} "
            f"q1.grad={grad_norms['q1']} "
            f"q2.grad={grad_norms['q2']}"
        )

    except ImportError as e:
        RESULT["error"] = f"MISSING_DEP: {e}"
        print(f"SKIP: flash-linear-attention nije instaliran ({e})")
    except Exception as e:
        RESULT["error"] = str(e)
        print(f"FAIL: neocekivana greska: {e}")
        raise


if __name__ == "__main__":
    _run_check()
