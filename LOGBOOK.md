# 📓 NHẬT KÝ DỰ ÁN — LE-WM → Hybrid Mamba-2+Attention

> **Dự án:** World model cho robot — từ tay bionic 8 khớp tự chế đến
> kiến trúc Hybrid Mamba-2+Attention và robot nhặt rác di động.
> **Bản này:** nhật ký kể chuyện, rút gọn từ `plan/project_logbook.md`
> (bản đầy đủ giữ tại máy — số liệu ở đây đều lấy nguyên văn từ đó).
>
> **Hành trình trong một câu:** V0 tay robot thật nắm được chai → V1 thất bại
> có giá trị (CfC + SIGReg ≠ nhau) → V2.1 Mamba-2 thắng Push-T 94.7% →
> V2.5.2 robot nhặt rác $140 (đang làm).

---

## Bản thân dự án - nhìn nhanh

| Bản | Trạng thái | Kết quả chính |
|---|---|---|
| V0 | đã xong | Robot thật, tay bionic 8 khớp — nắm thật 100%; CfC độ trôi 0.000014/bước, 34× thấp hơn AR |
| V1 | bỏ | Hybrid CfC+Attention, TwoRoom 78% → 6% — nhiễu SIGReg × ODE khuếch đại |
| V2.1 | đã xong | Push-T 94.7% ± 3.1% (beat LeWM 86.0% ± 4.0%, +8.7%); TwoRoom 85.3% |
| V2.5.1 | tạm gác | EEZYbotARM fixed (tay gắn cố định) (tay gắn cố định) — code viết 50% thì chuyển hướng |
| V2.5.2 | **đang làm** | Robot nhặt rác di động, CfC-habit + OOD gate + SI, Nav2 + ROS2 |
| V2.9.x | đã xong | Neuroevolution 2 genome — research nền tảng |

---

## V0 — Tay robot thật (7/6 → 18/6)

### 07/06 — Scheduled Sampling cứu CfC (0.072 → 0.002)

Không phải tự nhiên, hôm nay là ngày tôi ức. Mấy chục lần nhìn lại số
0.072 trong cuộn dự đoán, trong khi AR là 0.0012 — cách nhau hơn 60 lần.

Sau mấy ngày CfC đỏ mặt khi cuộn dự đoán, tôi ngồi đọc lại bài báo CfC
và tự hỏi: thầy
Hasani bảo *"CfCs might express vanishing gradient problems for
long-term dependencies"* — vậy chỉnh lại quá trình dạy bằng Lấy mẫu theo
lịch trình (Scheduled Sampling) thì sao? Kỹ thuật từng cứu chuỗi dài cho
RNN cũ, không ai thử cho ODE-RNN trước.

Nhập ba dòng, huấn luyện lại, đánh giá: 0.072 → 0.002 sạch tự nhiên.
Số 0.002 lần đầu chạm AR (khoảng cách 1.82× thay vì 61×) — cải thiện 33.69×.
Và tôi nhận ra điều theo mình cả dự án: **CfC không yếu — nó bị dạy
sai từ đầu, do thiếu tái giảng theo lịch trình.**

> Dữ liệu: V3 0.072417 ± 0.001 (61× AR) → V4 0.002149 ± 0.0008 (1.82×
> AR); AR 0.001182 ± 0.0002; best epoch V4 = 30; teacher loss V4 ≈ 0.0012
> ngang AR. Chưa có bài báo nào áp dụng SS cho mạng liên tục trước đây —
> đây là điểm lạ đầu tiên của dự án.

Bài học: kiến trúc chỉ là nửa câu chuyện — cách dạy còn lại nửa kia.
Ngày mai: ablation LayerNorm → batch → tỉ lệ lịch trình.

### 08/06 — Bốn bài kiểm tra T1-T4: CfC đúng ở chỗ nào?

Một ngày chạy bốn phép thử với cả CfC lẫn AR:

| Phép thử | Kết quả | Nghĩa gì |
|---|---|---|
| T1 — cuộn dài 20 bước | CfC 0.0217 tổng, độ trôi 0.000014/bước vs AR 0.0719, 0.000481 | CfC 34× ổn định hơn — sân nhà thật |
| T2 — Δt thay đổi | Δt=8 (533ms): CfC 0.0007 vs AR 0.0013 — cải thiện 83% từ Δt=1 | CfC biết thời gian, AR thì không |
| T3 — hành động xa phân phối (×2) | CfC hụt 26×, AR chỉ 1.7× | AR bền hơn 15× — sân nhà AR |
| T4 — học tắt (inverse model) | 29× tệ hơn so với dự đoán trung bình | Vectơ tiềm ẩn chứa trạng thái, không chứa hành động — thiết kế JEPA đúng |

Kết chung: **CfC giỏi về mặt thời gian, AR giỏi chịu hành động xa phân
phối. Hybrid sẽ làm cả hai** — đây là lý do V1 ra đời.
(Đo thêm: CfC chạy 0.69ms vs AR 2.03ms — nhanh 3×.)

### 09/06 — Ngày tay robot thật nắm được chai

Tôi vẫn nhớ log đó: *"Grasp detected: position error S2,S4,S7 < 100"*
— 3/3 xác nhận — tay bionic 8 khớp tự chế nắm chai nước trong hộp thu dữ liệu. Đây là lần đầu đoạn phần cứng thật chạy trọn vẹn từ ảnh camera đến hành động.

Đi theo nó là một loạt thật:
- phát hiện nắm bằng sai số vị trí `|lệnh − thực tế| < 100` — đơn giản
  mà tin cậy hơn nhiều so với đọc lực (servo 4 idle = 1144 do đối kháng,
  load 0-2024 bị nhiễu)
- CEM 5×100×5 chạy 2-3s/bước (8× nhanh hơn trước)
- camera live khác dữ liệu huấn luyện: chi phí mô hình thật 2.8-3.5 vs tưởng tượng
  0.002-0.005 — **khoảng cách encoder** hôm sau phải giải thích

Nhìn lại, "grasp" là bài toán ác nhất cho JEPA: thay đổi thị giác chỉ vài
mm mà SIGReg coi là nhiễu. Bài học: phát hiện xong bằng tín hiệu phần cứng
(vị trí), không tin chi phí mô hình.

### 10/06 — Ngày toàn lỗi: 11 sai lầm, 0 checkpoint dùng được

Một trong những ngày "chán" nhất: 11 sai lầm trong một ngày —
tự bịa ngưỡng chi phí mô hình 0.5 (cái này mới tội), ghi đè file goal của mình,
chạy cell chưa mount Drive, path `data/` vs `data_/`, cài torchvision
trên Colab làm hỏng torch (reset runtime 5 lần), SIGReg sai shape
`(B*T,D)` thay vì `(T,B,D)`...

0 tệp trọng số dùng được sau sờ soạn cả buổi. Rút ra mấy điều xương máu:
- Colab có torch sẵn — chỉ cài `h5py ncps --no-deps`
- factory reset runtime sau mỗi lần cài nhầm
- viết 1 cell duy nhất: mount → augment → train — không rườm rà

Khép ngày: stable-worldmodel 0.1.1 docs ghi hỗ trợ `format=hdf5` nhưng
code không có (format.py:60 chỉ lance/folder/lerobot/video) — phải dùng
GPUDataset + script tự viết. Bài học: **đừng tin tài liệu, mở code ra xem.**

> Điểm sáng: sau fix SIGReg shape + augment theo chuỗi + bỏ background
> replace → mất mát 0.368 (vs 1.43 trước), 41s/epoch.

### 13/06 — Sửa resume, một lỗi ngớ ngẩn tốn $3

Điều đau nhất tuần: `xxx_weights.ckpt` **không bao giờ được tạo**. File
checkpoint của Lightning đặt tên theo epoch, còn script tôi tìm đúng cái
tên không tồn tại → mỗi lần restart là train lại từ đầu. Cả ngày quét
tham số λ trên Colab cộng ~$3 Vast đều bốc hơi theo cách đó.

Fix trong `train.py` + `module.py`:
- thêm ModelCheckpoint filename theo epoch, resume bằng `glob *.ckpt`
  lấy mới nhất
- hidden state CfC bị reset mỗi khung hình đánh giá → thêm `_carry_mode` chỉ carry
  trong cuộn dự đoán
- HF path hardcode → dùng `self.subdir` + `run_name`

> Ngày đó tôi chốt quy trình chuẩn từ nay: **Colab fix bug trước, Vast
> chạy clean** — T4 miễn phí, notebook trực quan, xong mới trả tiền GPU.
> Chi phí phiên vừa rồi: ~6 lần tạo-hủy × ~$0.5 = ~$3 + 1 giờ chờ boot.

### 15/06 — "CfC không yếu, tôi hiểu sai nó"

Hôm nay là ngày tôi tự sửa lại cả một niềm tin:
- "Mamba nhanh hơn CfC" — **tự bịa**. CfC là công thức dạng đóng, không cần
  bộ giải số, độ phức tạp O(K̃) chứ không phải O(T·d²). Cả hai đều O(T).
  Mamba lợi thế ở độ ổn định cuộn dự đoán, không phải tốc độ.
- "T càng lớn càng tốt" — T=16 quá mức cần thiết cho TwoRoom 2 bậc tự do.
  **Chốt T=4 cho mọi kiến trúc** để so sánh công bằng với LeWM.
- "Attention lọc nhiễu" — chưa chứng minh được, cần kiểm.

Quy trình 3 trụ cột: Lý thuyết → Bài báo/Số liệu → Thực nghiệm. Thiếu 1 trong 3
= "chưa biết, cần kiểm." Cái này thành luật sống của dự án.

### 17-18/06 — V2.1 Mamba-2: 14× cải thiện

Sau V1 (CfC) chỉ 78% ở ngân sách 50 và 6% ở ngân sách 150, nguyên nhân đã rõ:
**nhiễu SIGReg tích lũy qua trạng thái ẩn ODE** — CfC không tương thích
với SIGReg, không phải CfC kém.

Nên tôi thay predictor bằng Mamba-2 (Cấu hình C: heads=16, d_state=256,
expand=4, depth=6, T=4, batch=128, lr=5e-5, mô hình 16.6M — predictor
9.36M, encoder ViT-tiny 5.5M):

- TwoRoom: 86% (43/50) — ngang LeWM 87% với lỗi xấp xỉ, nhưng đây là
  **14× so với CfC 6%** cùng điều kiện ngân sách=150
- Loss cuối: val/pred_loss 0.00724
- Chi phí phiên này: ~$0.55 (3h10 RTX 5080 giá $0.175/h)

Rồi ngày 20/06 là đêm chốt: **Push-T 94.7% ± 3.1% (92, 98, 94)** trên
RTX 5090 và T4 — tái lập trên 2 GPU. LeWM chính thức: 86.0% ± 4.0% (T4),
88.0% ± 4.0% (5090). Khoảng cách +8.7% (T4) và +6.7% (5090), cùng 3 hạt
giống, cùng eval.py.

Số bài báo LeWM là 96% ± 2.83% — khoảng tin cậy chồng lấp với chúng tôi,
nên nói công bằng: "không thua kém, và tái lập ổn định trên 2 GPU."

> Đã lâu rồi mặt tôi mới ngồi ghi lại từng số: 92, 98, 94; 94, 98, 92.
> Hai lần, hai máy, cùng con số 94.7 — không phụ thuộc phần cứng.

---

## V2.9.x — Nghiên cứu nền tảng (21/06 → 25/06)

### 24/06 — Valley of death: max 37-47, mean 33

Chạy neuroevolution 2 genome (một genome chính 100×8 tham số qua CPPN,
một genome dopamine 5 số) qua 1000 thế hệ: fitness kẹp ở **37-47 (max),
33 (mean)** — không bao giờ chạm trần 50 của lý thuyết.

Lúc đầu tôi tưởng bug. Hoá ra không phải: CPPN thuần không thể tự sinh
hành động "đứng im" — con số đó là giới hạn cấu trúc, là điểm dữ liệu
thật của "Valley", không nên sửa môi trường hay thêm phần thưởng để che.
**Đúng những gì dự án này cần ghi nhận: chạy được 200 thế hệ × 128 cá thể
trong 11.4 phút trên T4 — khung JAX tốc độ được, chỉ là rào cản sinh học
tự nó hiện ra.**

Chi tiết thú vị: dopamine phân hóa (gradient 0.12 / hebbian 0.67 / GA 0.21
ở thế hệ 20) — cơ chế trộn tự thích nghi. Giữ làm nền nghiên cứu cho
tương lai (không đưa vào robot 5.2 vì vai trò khác).

### 25/06 — Bản chạy chính thức + 6 quy tắc JAX

Đưa lên bản chạy chính thức: MAX_GENES 200, thêm `--run_id`, checkpoint lên HF mỗi
500 thế hệ. Hôm đó cũng là ngày "lòng vòng JAX" — 6 quy tắc khắc cốt ghi
tim (không `if` trên mảng được trace, không `int()`/`float()` trên biến
traced, flags phải là closure, carry của `lax.scan` khớp kiểu, không slice
động, dùng mảng đầu ra thay scalar cho tracking).

Nhìn lại mã nguồn: ~1040 dòng, 8 file, 7 genome, 12 cell chạy qua, 23 bug
đã ghi. Đóng mục này với cái thật lòng: **GA chậm hơn RL 50-250× trên
Ant — no free lunch.** Nhưng khung hai-genome là thứ sau này tái dùng
cho robot nhặt rác.

---

## V2.5 — Sim arm (29/06 → 13/07, tạm gác)

### 03/07 — URDF scale bug — con robot "1.5 mét"

Hôm đó tôi nhìn sim mà cười: STL vẽ bằng **cm**, joint origin cũng cm
chứ không phải mét — robot trong sim cao 1.5m thay vì 18-30cm như thiết kế.

Fix: scale ×0.1, đo lại bằng thước thật: L2=92mm, L3=80mm, H=63mm. Đo IK
bằng thước kẹp, không tin tưởng số giấy.

### 05/07 — V2.5 chốt + 3 novelty

Architecture Mamba đầy đủ trên vi điều khiển giá $10 — mới là tuyên bố của
phiên này. Chốt: Vim + Mamba-2 với **độ chính xác lai (INT8 + FP32 state)**
— chưa ai công bố mô hình trạng thái (SSM) kiểu này trên vi điều khiển.
Kèm kiến trúc deploy: EKF + SLAM Toolbox + 2 nhánh camera. (Sau này hướng
này vẫn trụ — mô hình thật đi vào robot rác là world model một bước, nhưng
tinh thần vẫn nguyên: mô hình thật nhỏ, chạy được trên vi điều khiển.)

---

## V2.5.2 — Robot nhặt rác di động (30/07 → 01/08, đang làm)

### 30/07 — Bắt đầu: từ 6 tuần, tôi muốn robot chạy thật

Cuộc họp chốt phiên 12h rồi chạy tiếp đến 14h: **robot nhặt rác di động**
— dự kiến 6 tuần (30/07 → 09/09). Khung quyết định lần này:
- Nav2 + ROS2 ngay từ đầu, không encoder (chỉ LiDAR + IMU EKF + KLT),
  không sim nữa — thu 500 tập thật
- Hành động 5D [vx, ω, S1, S2, grip]
- Ngân sách: LiDAR $25 + IMU $3 + chassis $15 + pin $10 = $53
- Novelty: IK + world model, CL với OOD_ema thích nghi, metric thời gian
- Trước đó tôi suốt buổi V2.5.1 (EEZYbotARM fixed) — nhìn lại 6 tuần là
  không đủ cho tay cứng, phải rút mỏng kiến trúc để kịp vòng đua:

### 31/07 2h sáng — 14 giờ phản biện mới chốt được spec

Đêm đó tôi còn nhớ: 4h chiều 30/07 đến 6h sáng 31/07 — 14 giờ liên tục
phản biện. Ra spec đầu tiên. Nhưng đủ mùi "tham lam":
- Mamba multi-step (Vim + Mamba-2 + DCEM + action decoder 9.36M)
- hàm lỗi 4 thành phần, trọng số theo độ bất định (Kendall)
- z_goal học được tự do

Tôi biết nó đẹp nhưng không tin nó chạy ở đây. (Hôm sau 01/08 chứng
minh tôi đúng — xem mục dưới.)

Chi tiết đáng nhớ: robot_localization EKF 3 sensor (rf2o LiDAR ~0.7,
KLT ~0.2, IMU ~0.1 trọng số), map duy nhất từ SLAM Toolbox, camera forward
720p thay 1080p (vì USB 2.0 90MB/s > 60MB/s), SLAM dừng trong ±2.5cm.

### 31/07 — 6 giờ dài: nhìn ra những lỗi đọc tài liệu tự mình

Một ngày vừa rồi trong "mỗi 1 giờ 1 quyết định" là ngày tôi thấy mình
đọc tài liệu theo kiểu muốn một kết luận cho sẵn:

1. **Sai thứ nhất — pin:** "Pin sag 11.1→9V" tôi viết — sai. Samsung 20R có nội
   trở ~40mΩ, stall 13.3A chỉ gục 0.5V. Thật: 10.5V.
2. **Điện năng xem thường.** "Servo idle 0.3W" — sai luôn. Holding của
   mỗi con là 0.2-0.5A → 3 con ≈ **4.5W** liên tục, thiếu 15×.
3. **"Không có OOD gate ngoài bài báo!"** — sai. Science Robotics 2024
   (Vorbach, Hasani) test CfC với visual OOD rồi.

Cái nào cũng khiến tôi tự đặt lại câu "tôi biết hay tôi muốn đúng?"

Từ 6h-7h: cấu trúc 4 lớp cảm giác cho robot (mắt + mắt cận + cảm giác
bản thân về góc khớp + cảm giác chạm) — cái này kết nối trực tiếp thế
giới đọc hiểu của robot vào phần thích kiểm tra thực tế.
Từ 7h-8h: z_goal vòng luẩn quẩn — bỏ học tự do, chốt **centroid thống kê
từ offline các frames thành công, và FREEZE nó**.

Từ 8h-9h: hàm lỗi 4 thành phần có xung đột giữa MobileNet head → giải bằng
SI weight-distance (bài học CVPR "Mind the Backbone") + action decoder
MLP 1K + trọng số theo độ bất định (Kendall) thay số heuristic 0.3/0.05 bịa.
Từ 9h-10h: camera 720p + 2 nhánh resize.
Từ 10h-12h: 2 hộp pin 3S, **star ground bắt buộc** ở GND buck, khối logic
5V (S905X3 2-5W + LiDAR 1-2W) hóa ra lại là chỗ ăn điện nhiều hơn motor —
mẫu chuyện điện này còn theo suốt phần vận hành.

### 01/08 — Pivot toàn bộ V2.5.2: bỏ Mamba multi-step, giữ world model một bước

Hôm đó là ngày tôi nhặt lại túi đồ từ bảng đề bài và đọc lại kết quả
nghiên cứu: nếu model "multi-step WM" vốn vẫn đứng đó, thì còn gì đứng
lại được với máy tính nhỏ?

Nên quyết định gạch:
- ❌ Mamba-2/Vim/DCEM — research 2026 nói "chưa ai vượt được giới hạn
  tầm nhìn ngắn của world model" (LeWM, WorldPlanner) — tôi không cần
  làm lại con đường từng chết
- ✅ world model một bước = CfC-imagination — đủ làm bộ kiểm chứng
  (thử ~20-30 lệnh quanh lệnh đề xuất, chọn lệnh gần tâm thành công nhất)
- ✅ CfC-habit (residual policy 20-75K), học từ 500 tập thao tác
- ✅ OOD gate z − z_goal với α = sigmoid((OOD−1.5)×2)
- ✅ SI (λ Σω(W−W₀)²) giữ trí nhớ; còn z_goal, σ, analytic, backbone
  đóng băng giữ nguyên trong suốt quá trình học

"V2.5.2" giờ có 4 cụm ý chính: **reflex + tự biết lạ + học an toàn +
tưởng tượng (bộ kiểm chứng)**. Kiến trúc gọn: MobileNetV2
+ homing 3 hằng số → 2 mạng CfC với OOD → học 2 kênh (tự động + teleop).

Ba lỗi tôi tự bắt đêm 31/07 (pin, servo holding, OOD bài báo) đều được ghi
thẳng vào bảng minh bạch trong spec. Có chút xấu hổ, nhưng
xấu hổ đúng chỗ còn hơn tự tin sai.

---

## Phần cứng hóa — 02/08 → 03/08

### 02/08 — "opencode chết = tôi chết" — DNS tĩnh cứu phiên

Đúng là "network saga" — bài học xương máu:
- WiFi AP của robot (Realtek 8822bs) mặc định dùng DNS nhà cung cấp
  chặn domain `opencode.ai` — mà vì tôi sống bằng MCP, không fix nổi
  mạng thì không làm được gì, không một phiên nào chạy nổi
- fix: DNS tĩnh 8.8.8.8 / 1.1.1.1 cho adapter. Sau đó: AP channel vì
  driver vendor không nhận từ nl80211 → `modprobe 8822bs
  rtw_ips_mode=0 rtw_power_mgnt=0 rtw_channel=6` + reboot.
- dnsmasq standalone (dhcp-range 10.0.0.10-100) + NAT (ip_forward +
  MASQUERADE) — vì chia sẻ mạng của nmcli phụ thuộc route mặc định,
  không ổn định cho AP robot.
- Kết: box `curl -x http://127.0.0.1:3128 https://...` = 200.
- Bài học treo tường: mọi thao tác cuối đều phải qua box — nên phần
  mạng phải sẵn sàng TRƯỚC mỗi phiên chạy, không phải lúc chạy mới hỏi.

### 03/08 — N5 Max lên OS đầy đủ: 10 lệnh USB boot + WiFi dựng từ driver vendor

Ngày "hoàn thành phần cứng mạng":
- 10 lượt script lẫn nhau (hết pin laptop, đĩa offline, volume E, quyền
  access denied...). Lượt 9: ghi bảng phân vùng xong xuôi nhưng quên
  seek → lệch 8MB toàn bộ data; may bắt được bằng verify. Lượt 10: fix seek,
  verify đầu + cuối 100%.
- kernel: 6.12 crash, rút về 5.15.211 + air dtb.
- WiFi build từ vendor cho Realtek 8822bs: thiếu prefix `usr/src`,
  `syncconfig` glibc 2.38 vs box 2.36 → xóa autoconf.h, modpost compile
  không `-I kernel`... Kết: `8822bs.ko 5.5MB → modprobe OK → wlan0 UP`.
- SSH: `systemctl enable --now ssh`.
- Máy gần như lên mạng đầy đủ lần đầu, từ OS trở lên.

> Buổi chiều: test phần cứng ESP32-S3 + IMU + LiDAR. IMU pass (0x68,
> ~1.06g, bias gx≈3300). LiDAR Camsense X1 frame 36 bytes (55 AA 03 08 |
> RPM u16/64 ... | 8×(dist u16LE mm + intensity) | CRC; intensity 0 = no
> reading; RPM ≈310). LiDAR cần 5V riêng — USB laptop không đủ. S3 COM8
> CH343 → box ttyACM0.
> Bug nhớ: tắt nguồn không sạch → chip WiFi rác ("rtl8822b_power_off:
> Power OFF Fail!!") → `modprobe -r 8822bs && modprobe 8822bs` hồi sinh.
> LM2596S 3A lúc đó thật ra chỉ 2A, chạy nóng 90-110°C — đổi XL4015 5A ×2.

---

## V2.5.1 → V2.5.2 — dọn dẹp (18/07 → 22/07 tóm tắt)

- 18/07: **`git reset HEAD~1 --hard` xóa 6 giờ code chưa commit** —
  giáo dạy theo cách đau: `--soft` giữ file, `--hard` xóa, `git add <file>`.
- 22/07: bug calib auto-sort swap direction (lỗi nối nhau thành 4 tầng) —
  nhìn lại bài học bật nhất: **sự thật vật lý > số học thuận tiện** –
  nghĩ ra phải kiểm chứng theo logic vật lý thực tế.
- 22/07: quyết định tham chiếu hình học thật: L2=92mm L3=80mm H=63mm (kẹp),
  bỏ K coupling, giữ K=0.55 stretch 68.8°→180°.

---

## V0-2.1 — số liệu chính (bảng tra nhanh)

| Thứ | Số | So với |
|---|---|---|
| Push-T | 94.7% ± 3.1% | LeWM 86.0% ± 4.0% (T4), +8.7% |
| Push-T (5090) | 94.7% ± 3.1% (94,98,92) | LeWM 88.0% ± 4.0%, +6.7% |
| TwoRoom | 85.3% ± 10.1% (84,76,96) | LeWM 80.7% ± 10.3% — chồng lấp |
| TwoRoom (5090) | 86.0% ± 10.0% | LeWM 81.3% ± 9.5% |
| CEM thời gian | ~85s/lần | LeWM ~20s (T4) — chậm ~4× |
| CfC V0 độ trôi | 0.000014/bước | AR 0.000481 — 34× |
| Lấy mẫu theo lịch trình | 0.072 → 0.0025 | 29× cải thiện |

---

## Bài học cứng — chọn ~15 điều đáng nhớ nhất

1. **3 trụ cột: Lý thuyết + Bài báo/Số liệu + Thực nghiệm** — thiếu 1 thứ = "chưa biết".
2. **Đừng tự bịa ngưỡng/tốc độ/kết luận** — số chưa đo thì viết "chưa rõ".
3. **Colab sửa lỗi trước, Vast chạy sạch** — T4 miễn phí, không đốt tiền chờ máy.
4. **Check state_dict keys trước khi trả lời đánh giá** — 1 thay đổi nhỏ trong
   encoder làm hỏng toàn bộ trọng số cũ.
5. **Lấy mẫu theo lịch trình là can thiệp hiệu quả nhất cho ODE-RNN liên tục.**
6. **CfC ≠ yếu temporal, CfC yếu OOD action** — xoay thiết kế quanh 2 sân này.
7. **Làm "kiến trúc vạn năng" nhiều khi là dấu hiệu của niềm tin không kiểm chứng** — chốt T=4 cho mọi kiến trúc.
8. **Bài báo là input cho thiết kế, không phải bằng chứng cho đúng.** Dự án này 2 lần "bịa" vì đọc bài báo theo kiểu "muốn" — pin sag, servo watt.
9. **V0/V1/V2.1 số khác nhau, đừng gộp** — LeWM bài báo có số riêng, V0 fork có số riêng.
10. **Thời gian sửa lỗi trên Colab/Kaggle: giới hạn 5 giờ/phiên** — chia phiên theo giới hạn đó.
11. **Commit chỉ khi đủ dữ liệu; message ghi WHAT không ghi VERDICT** — "đã hay tốt" không kiểm chứng được.
12. **Đừng xóa kiến trúc cũ bằng "không cần"** — phân định rõ bỏ (= dừng), gác (= chuyển hướng trong khi còn nguyên bản mẹ).
13. **Suppress vs eliminate** — tham số che dấu chứng, sửa mới triệt để cần thay đổi cơ chế.
14. **Giải thích lại như dạy người mới (3 buổi Feynman) mỗi phiên lớn** — người đọc xa lạ hiểu đúng, không hiểu nhầm.
15. **Đừng tối ưu tham số, hãy so sánh kiến trúc công bằng** — bỏ vòng "tìm T tốt", đó là lãng phí giờ máy.

---


