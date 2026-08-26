# Nghiên cứu kiến trúc Hybrid ODE CfC-Attention cho World Model trong robot manipulation

---

## Mục lục

1. [Mở đầu](#1-mở-đầu)
2. [Cơ sở lý thuyết](#2-cơ-sở-lý-thuyết)
3. [Thực nghiệm trên robot thật (V0)](#3-thực-nghiệm-trên-robot-thật-v0)
4. [Kiến trúc Hybrid ODE CfC-Attention (V1)](#4-kiến-trúc-hybrid-ode-cfc-attention-v1)
5. [Thực nghiệm mô phỏng TwoRoom](#5-thực-nghiệm-mô-phỏng-tworoom)
6. [Kết quả và thảo luận](#6-kết-quả-và-thảo-luận)
7. [Kết luận và hướng phát triển](#7-kết-luận-và-hướng-phát-triển)
8. [Tài liệu tham khảo](#8-tài-liệu-tham-khảo)

---

## 1. Mở đầu

### 1.1. Bối cảnh

Robot manipulation — khả năng tương tác vật lý với môi trường — đòi hỏi mô hình có thể dự đoán hậu quả của hành động trong không gian trạng thái phức tạp. Các phương pháp điều khiển cổ điển (analytical model) không linh hoạt với môi trường mới. Học tăng cường (reinforcement learning) yêu cầu hàng triệu episode tương tác thực tế — chi phí không khả thi cho doanh nghiệp vừa và nhỏ.

World model (mô hình thế giới) là hướng tiếp cận dung hòa: học mô hình dự đoán trạng thái từ dữ liệu offline, sau đó dùng bộ lập kế hoạch (CEM) để chọn hành động tối ưu trong không gian tưởng tượng.

### 1.2. Vấn đề

LeWM (Maes et al. 2026) là world model JEPA sử dụng bộ dự đoán AR (Autoregressive Transformer) với hai hàm mất mát duy nhất: sai số dự đoạn MSE + điều hòa SIGReg (λ=0.09). Cấu hình: T=4 (ba khung hình ngữ cảnh, một dự đoán), batch=128. Đạt kết quả cạnh tranh trên bốn bài kiểm chuẩn: TwoRoom 87%, Push-T 96%, Cube 74%, Reacher 88%.

AR có hạn chế: sai số tích lũy trong dự đoán chuỗi dài. ODE CfC (Hasani et al. 2022) — biến thể ODE-RNN với trạng thái ẩn liên tục — hứa hẹn giảm sai số tích lũy. Chúng tôi đề xuất thay AR bằng ODE CfC kết hợp với cơ chế chú ý (Self-Attention).

**Qua thử nghiệm V1, chúng tôi phát hiện:** SIGReg noise tương tác với CfC hidden state, gây tích lũy lỗi qua các bước thời gian. Đây là hướng cải thiện chính cho V1.1.

### 1.3. Mục tiêu

1. Xây dựng kiến trúc Hybrid ODE CfC-Attention cho world model.
2. Đánh giá trên robot thật (tay bionic 8-DOF) và mô phỏng TwoRoom.
3. So sánh công bằng với AR baseline ở cùng T=4, batch=128.

### 1.4. Đóng góp

1. Kiến trúc Hybrid ODE CfC-Attention.
2. Scheduled sampling trên ODE CfC — lần đầu cho world model.
3. Pipeline thu thập dữ liệu có kiểm soát domain gap.
4. **Phát hiện:** SIGReg noise tương tác với CfC hidden state — nguyên nhân Hybrid (T=16) chỉ đạt 78%.
5. Mã nguồn mở + logbook.

---

## 2. Cơ sở lý thuyết

### 2.1. JEPA

JEPA (LeCun 2022): encoder biến đổi pixel → latent embedding, predictor dự đoán embedding tương lai. Loss: MSE + SIGReg. Không tái tạo pixel.

### 2.2. ODE CfC

ODE CfC (Hasani et al. 2022) có hidden state biến đổi liên tục theo ODE:
```
h(t+Δt) = h(t) + f(h(t), x(t)) · Δt
```
Duy trì hidden state qua nhiều bước với sai số thấp. Nhưng hidden state ODE nhạy với nhiễu ở input — khuếch đại qua các bước.

### 2.3. LeWM paper — AR predictor

AR predictor: Transformer 6 layer, 16 heads, AdaLN. T=4 cố định cho cả 4 benchmark. Encoder: TinyViT 12.3M. SIGReg (λ=0.09, knots=17, num_proj=1024).

### 2.4. SIGReg

SIGReg đo Epps-Pulley statistic trên chiếu ngẫu nhiên → ép embedding Gaussian. Tạo nhiễu ngẫu nhiên trên embedding (do random projections). Với AR: nhiễu không tích lũy. Với ODE CfC: nhiễu khuếch đại qua hidden state.

### 2.5. Scheduled Sampling

Bengio et al. (2015): thay ground-truth bằng prediction với xác suất p giảm dần. Quan trọng với ODE CfC vì hidden state carry khuếch đại sai số từ bước trước.

---

## 3. Thực nghiệm trên robot thật (V0)

### 3.1. Mục đích

Xác nhận pipeline world model + CEM hoạt động trên phần cứng thật, so sánh rollout drift CfC vs AR.

### 3.2. Phần cứng

Tay bionic 8-DOF (DexHand V1, open source), 3 ngón đối kháng. Cải tiến: lò xo duỗi ngón. Khung in 3D (PLA).

<p align="center">
  <img src="robot-neutral.jpg" width="70%">
  <br><em>Robot tay bionic 8-DOF — vị trí neutral</em>
</p>

<p align="center">
  <img src="grasp-luc-thu-data.jpg" width="80%">
  <br><em>Box thu dữ liệu — camera cố định, lighting chuẩn, background đồng nhất</em>
</p>

| Linh kiện | Model | Thông số |
|---|---|---|
| Servo | SC09 (Waveshare) | Bus SCS CL, 0-1023, 300°, 1A@6V |
| Số lượng | 8 | 3 ngón × 2-3 servo/ngón |
| Chuyển đổi | USB-UART adapter | UART ↔ USB + cấp nguồn servo |
| Nguồn servo | 3×18650 20A | Qua mạch hạ áp → 6V |
| Camera | Webcam USB | 480p, CAP_DSHOW, crop 364×364 |

<p align="center">
  <img src="so-do-mach.png" width="90%">
  <br><em>Sơ đồ kết nối phần cứng</em>
</p>

<p align="center">
  <img src="sc09.jpg" width="20%">
  <img src="usbto-serial.jpg" width="25%">
  <img src="web-cam-480p.jpg" width="20%">
  <img src="hop-3cell.jpg" width="20%">
  <img src="pin-cell-samsum.jpg" width="20%">
  <br><em>Linh kiện: servo SC09, USB-UART adapter, webcam 480p, hộp 3 cell 18650, pin Samsung</em>
</p>

### 3.3. Thu thập dữ liệu

Box kín với camera cố định, đèn LED fixed exposure, background trắng. ~50 episode neutral→grasp, ~10,000 frame. Augment ColorJitter → 17,800 frame. Phát hiện grasp: position error |cmd-actual|<100 trên 3 servo chính.

### 3.4. Kết quả V0

| Model | Prediction loss (train) | Prediction loss (val) | Rollout drift (step 10) |
|---|---|---|---|
| AR | 0.0013 | 0.0022 | 0.00048/step |
| CfC (V4) | 0.0052 | 0.0004 | **0.000014/step** |

CfC drift thấp hơn AR ~34×. Grasp thành công với chai nước.

### 3.5. Hạn chế phát hiện

CfC yếu với OOD action — motivation cho Hybrid: Attention làm spatial, CfC temporal, giảm OOD gánh cho CfC.

---

## 4. Kiến trúc Hybrid ODE CfC-Attention (V1)

### 4.1. Mô tả

Kiến trúc kế thừa JEPA từ LeWM paper, thay AR predictor bằng 6 block Hybrid.

<p align="center">
  <img src="hybrid_architecture.png" width="70%">
  <br><em>Kiến trúc Hybrid ODE CfC-Attention — từ input đến predicted embedding</em>
</p>

Mỗi block:
- **Attention:** heads=16, dim_head=64 → 787K params
- **ODE CfC:** backbone_units=384, cfc_hidden=256 → 764K params
- **Tỉ lệ CfC:Attention ≈ 1:1**

| Component | Params/block | Tổng (6 blocks) |
|---|---|---|
| Attention | 787K | 4.72M |
| ODE CfC | 764K | 4.58M |
| AdaLN | 222K | 1.33M |
| **Predictor total** | | **~10.6M** |

### 4.2. Denoiser MLP

Residual MLP giữa encoder và predictor, lọc SIGReg noise trước CfC.

### 4.3. Scheduled Sampling

Xác suất p = 1 - (epoch/max_epochs) thay ground-truth bằng prediction.

### 4.4. Thiết lập tham số

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| T | 4 | history_size=3, num_preds=1 |
| frameskip | 5 | |
| batch_size | 128 | |
| lr | 5e-5 | AdamW |
| sigreg weight | 0.09 | |
| seed | 3072 | |

**Lưu ý:** V1 đầu tiên thử nghiệm với T=16 để khai thác tối đa khả năng temporal của CfC. Qua đó phát hiện CfC nhạy với nhiễu SIGReg ở khung thời gian dài. **V1.1 (hiện tại) chốt T=4** theo đúng thiết lập paper.

---

## 5. Thực nghiệm mô phỏng TwoRoom

### 5.1. Môi trường

TwoRoom: hai phòng, cửa ở giữa. Robot cần nhớ đường đi qua cửa. Dữ liệu từ LeWM (89,000 frame).

### 5.2. Huấn luyện V1 (T=16)

- Cấu hình: **T=16** (history_size=15), heads=8, L40S (Vast.ai)
- 10 epochs, batch=128
- Thời gian: ~3h

**Phát hiện:** T=16 không tương thích với thiết lập LeWM paper (T=4). Đây là bài học về thiết kế thí nghiệm: so sánh kiến trúc phải cùng tham số.

### 5.3. Kết quả

**Kết quả chính (ngân sách hành động 50 bước — thiết lập mặc định):**
| Kiến trúc | T | TwoRoom |
|---|---|---|
| AR (LeWM paper) | 4 | **87%** |
| Hybrid V1 (gốc) | **16** | **78%** |
| Hybrid V1.1 | **4** | 🔄 đang chạy |

**Kết quả mở rộng (ngân sách 150 bước — theo đúng thiết lập LeWM paper):**
| Kiến trúc | T | TwoRoom |
|---|---|---|
| Hybrid V1 (gốc) | 16 | **6%** |
| Hybrid V1 (gốc) | **4** | **4%** |

Phát hiện quan trọng: khi tăng ngân sách hành động lên 150, CfC hidden state tích lũy nhiễu SIGReg đủ lớn để phá hỏng dự đoán. Kết quả 4-6% (so với 78% ở ngân sách 50) cho thấy SIGReg noise ảnh hưởng CfC theo cấp số nhân khi rollout dài. Đây là động lực chính cho V1.1 (denoiser + λ sweep) và V2 (Mamba — không có ODE hidden state).

---

## 6. Kết quả và thảo luận

### 6.1. Hai phát hiện chính

**1. T=16 không dẫn đến kết quả tốt hơn.** Trái với kỳ vọng ban đầu, CfC không tận dụng được context dài 15 khung hình. Lý do: SIGReg noise tích lũy qua từng bước ODE.

**2. SIGReg noise ảnh hưởng CfC mạnh hơn AR.** SIGReg (λ=0.09) tối ưu cho AR (không có trạng thái ẩn). Với ODE CfC, nhiễu từ các phép chiếu ngẫu nhiên của SIGReg khuếch đại qua động học ODE, thể hiện rõ nhất ở rollout dài (ngân sách 150: 4-6%).

V0 test chứng minh CfC drift 0.000014/bước, tốt hơn AR 34× — CfC không yếu. Vấn đề là **tương tác SIGReg-CfC.**

### 6.2. Bài học thiết kế thí nghiệm

Ban đầu chúng tôi chọn T=16 với kỳ vọng "càng nhiều ngữ cảnh càng tốt". Thực tế cho thấy: so sánh kiến trúc phải cùng tham số. **Cùng T, cùng batch, cùng bộ mã hóa. Chỉ khác bộ dự đoán.** Kiến trúc nào tốt hơn ở cùng điều kiện mới là cải tiến thực sự.

### 6.3. Giải pháp cho V1.1

1. **Denoiser MLP** — lọc SIGReg noise.
2. **λ sweep** — tìm λ phù hợp cho CfC.
3. **Scheduled sampling** — giảm exposure bias.
4. **BF16 precision** — giải quyết overflow SIGReg.

---

## 7. Kết luận và hướng phát triển

### 7.1. Kết luận

Kiến trúc Hybrid ODE CfC-Attention đã được xây dựng và kiểm chứng trên robot thật. Kết quả 78% TwoRoom (T=16) thấp hơn AR (T=4) 87% — nguyên nhân chính là CfC nhạy với nhiễu SIGReg. Phát hiện này mở ra hướng giải quyết: denoiser + λ sweep (V1.1) và thay thế CfC bằng Mamba (V2) — loại bỏ ODE hidden state, triệt tiêu tích lũy sai số khi dự đoán chuỗi dài.

### 7.2. Hướng phát triển

1. **V1.1:** Denoiser + λ sweep + T=4 → mục tiêu >87%.
2. **V2:** Mamba — triệt tiêu tích lũy sai số khi dự đoán chuỗi dài.
3. **Mở rộng xã hội:** Multi-robot, camera trên cao, biểu diễn chung.

### 7.3. Ứng dụng

**Hiện tại:** Robot tay gắp bionic 8-DOF. Chi phí phần cứng ~$100 (servo SC09, adapter USB-UART, pin 18650, mạch hạ áp, khung in 3D). Huấn luyện trên Colab free → $0.

**Tầm nhìn:** Robot nhà máy, hỗ trợ người khuyết tật. Kiến trúc nhẹ (~15M params), mã nguồn mở.

---

## 8. Tài liệu tham khảo

1. Maes, L. et al. (2026). LeWorldModel. *arXiv:2603.19312*.
2. Hasani, R. et al. (2022). Closed-form Continuous-time Neural Networks. *Nature Machine Intelligence*.
3. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence.
4. Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
5. Bengio, S. et al. (2015). Scheduled Sampling. *NeurIPS*.
6. Assran, M. et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. *CVPR*.

---

*Mã nguồn: https://github.com/thoan4965-ui/hybrid-cfc-atention-WM*
*Logbook: LOGBOOK.md trong repository*
*Checkpoint: https://huggingface.co/hhian/checkpoints*
