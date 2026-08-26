# Robot Bionic Hand 8-DOF — Hardware & Planning

Code real robot cho V0: bionic hand 8-DOF grasp planning via CEM + CfC/AR predictor.

## Thành phần

| File | Chức năng |
|---|---|
| `robot_planner.py` | CEM planner + MPC loop chính |
| `camera_goal.py` | Chụp goal image từ camera → encode → lưu |
| `serial_servo.py` | Giao tiếp SC09 servo qua scservo_sdk (COM13, 1Mbps) |
| `set_cam.py` | Căn chỉnh crop zone camera |
| `calibrate_grasp.py` | Calib vị trí grasp cho từng servo |
| `calib_tune.py` | Tune từng servo riêng lẻ |
| `augment_h5.py` | Augment dataset H5 (ColorJitter) |
| `convert_ckpt.py` | Convert checkpoint format |
| `enc_mse.py` | So sánh encoder latent giữa 2 ảnh |
| `ping_servos.py` | Kiểm tra kết nối servos |
| `test_encoder.py` | Test encoder robustness (lighting, background) |
| `test_color_goal.py` | Test color variance of goal |
| `test_current.py` | Đọc dòng servo |

## Hardware setup

- **Servo:** SC09 (SCS CL protocol), range [0, 1023]
- **Khung:** In 3D, 8-DOF (3 ngón: cái, trỏ, giữa)
- **Camera:** Webcam 480p, ID 1, CAP_DSHOW mode
- **Serial:** COM13, 1Mbps, scservo_sdk

## Pipeline

```
Camera capture → encoder → CEM plan → servo execute → position error check → done
```

## Tham khảo

- DexHand V1: https://github.com/TheRobotStudio/V1.0-Dexhand
- SC09 datasheet: https://www.waveshare.com/wiki/SC09_Servo
