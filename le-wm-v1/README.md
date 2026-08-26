# V1 — Hybrid CfC+Attention (Abandoned)

Hybrid CfC+Attention predictor cho TwoRoom. Đạt 78% (budget 50) nhưng giảm còn 6% (budget 150) do SIGReg noise × ODE hidden state.

**Không sử dụng cho benchmark chính — thay thế bởi V2.1 (Mamba-2+Attention).**

Giữ lại để tham khảo kiến trúc và so sánh lịch sử:
- `module.py` — HybridCfCPredictor (CfC + Attention)
- `jepa.py` — JEPA (khác với V2.1)
- `eval.py` — Eval TwoRoom
- `train.py` — Training
