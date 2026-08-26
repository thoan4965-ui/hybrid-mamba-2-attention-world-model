# Hybrid Mamba-2+Attention World Model cho robot manipulation

Nghiên cứu kiến trúc lai cho bộ dự đoán JEPA world model: thay MLP không trạng thái (LeWM AR) bằng **Mamba-2 trạng thái rời rạc** — vừa giữ lợi thế temporal, vừa không khuếch đại nhiễu.

**Kết quả chính:**

| Task | Hybrid Mamba-2 | LeWM AR (official) | Chênh lệch |
|---|---|---|---|
| Push-T (T4 fp32, 3 seeds) | **94.7% ± 3.1%** | 86.0% ± 4.0% | **+8.7%** |
| Push-T (RTX 5090, 3 seeds) | **94.7% ± 3.1%** | 88.0% ± 4.0% | **+6.7%** |
| TwoRoom (T4, 3 seeds) | 85.3% ± 10.1% | 80.7% ± 10.3% | +4.6% |

## Thư mục

| Thư mục | Nội dung |
|---|---|
| `le-wm-v2.1/` | **Code chính** — Hybrid Mamba-2+Attention (train/eval/module/jepa + config) |
| `le-wm-vo/` | **V0** — robot thật tay bionic 8-DOF + so sánh CfC vs AR + CEM planner |
| `le-wm-v1/` | **V1 (tham khảo)** — Hybrid CfC+Attention (abandoned — nguồn phát hiện SIGReg×ODE) |
| `baocao/` | Báo cáo thuyết minh cuộc thi (PDF + nguồn + hình) |

## Tái lập kết quả (3 bước — như trong báo cáo)

**Bước 1 — Vận hành phần cứng V0 (teleop cử chỉ):**
```bash
cd le-wm-vo/robot
python demo_grasp.py
```

**Bước 2 — World model + CEM grasp trên robot thật:**
```bash
python robot_planner.py --goal <file> --model cfc --checkpoint <ckpt>
```

**Bước 3 — Tái lập eval V2.1 (Push-T):**
```bash
cd le-wm-v2.1
python eval.py --config-name=pusht policy=<ckpt> --seed=3072
```

**Ghi chú:** môi trường huấn luyện/eval cần GPU (Linux + CUDA) và `mamba-ssm` (wheel PyPI). Checkpoint công khai tại [HuggingFace](https://huggingface.co/hhian/checkpoints).

## 3 đóng góp chính
1. **Scheduled Sampling cho CfC** — cải thiện chuỗi dự đoán 29× (lần đầu cho ODE-RNN/CfC).
2. **Phát hiện SIGReg × ODE** — trạng thái ODE khuếch đại nhiễu SIGReg (CfC 78%→6% goal xa).
3. **Kiến trúc lai block-level Mamba-2+Attention** — Attention + SSM trong cùng block (khác Jamba/TransMamba layer-level).

## Tham khảo
- [1] LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence.
- [2] Maes, L. et al. (2026). LeWorldModel. arXiv 2603.19312.
- [3] Hasani, R. et al. (2022). Closed-form Continuous-time Neural Networks. Nature Machine Intelligence.
- [4] Dao, T. & Gu, A. (2024). Transformers are SSMs. ICML 2024.
- [5] Balestriero, R. & LeCun, Y. (2025). LeJEPA. arXiv 2511.08544.
- [6] Ma, C. & Najarian, K. (2025). Rethinking the long-range dependency in Mamba/SSM and transformer models. arXiv 2509.04226.
- [7] Huang, Y. (2026). VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models. arXiv 2601.14354.
- [8] Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv 2312.00752.
- [9] Lieber, O. et al. (2024). Jamba: A Hybrid Transformer-Mamba Language Model.
- [10] Li, Y. et al. (2026). TransMamba: A Sequence-Level Hybrid Transformer-Mamba Language Model. AAAI 2026.
- [11] Liu, X. et al. (2020). Neural SDE: Stabilizing Neural ODE Networks with Stochasticity. NeurIPS 2020.
- [12] Rob Knight (2022). DexHand V1.0: Open-Source Dexterous Humanoid Robot Hand. GitHub.
- [13] MambaLite-Micro (2025). Mamba LLM trên MCU với INT4 quantization. arXiv 2509.05488.
- [14] Quamba: Post-training quantization INT8 cho Mamba/SSM triển khai edge.
