# -*- coding: utf-8 -*-
"""
demo_grasp.py — GRASP TELEOP MEDIAPIPE (real-time — 8 servo — cầm chai!)
3-4 GÓC (mỗi ngón riêng!): thumb_flex (1,2,3→1!) — thumb_opp (0,1,4→2!) —
flex_trỏ (5,6,7→4!) — flex_giữa (9,10,11→7!) — khép (6/9 — 153/218!) —
dạng (5/8 — giữ 0 — bị động!) + flex_max auto (3s — gập hết tay — 4 max!) +
"CẦM!" (load detect — 2/4/7!) — ESC thoát!
DATA: calib_neutral + calib_grasp + calib_dang (3 config — 2 mốc mỗi trục!)
"""
import sys
import os
import json
import time
import msvcrt

import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import serial_servo

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = r"C:\Users\duong\AppData\Local\Temp\opencode\hand_landmarker.task"
BASE = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(BASE, "..", "data", "calib")
NEUTRAL_PATH = os.path.join(CALIB_DIR, "calib_neutral.json")
GRASP_PATH = os.path.join(CALIB_DIR, "calib_grasp.json")
DANG_PATH = os.path.join(CALIB_DIR, "calib_dang.json")
LOAD_PATH = os.path.join(BASE, "load_threshold.json")
GRASP_CFG_PATH = os.path.join(BASE, "demo_grasp_config.json")

GRASP_SERVOS = [1, 2, 4, 5, 6, 7, 8, 9]
LOAD_DETECT_SERVOS = [2, 4, 7]
EMA_N = 5
SAMPLE_MS = 0.03


def angle_between(p1, midpt, p2, plane=None):
    ba = p1 - midpt
    bc = p2 - midpt
    if plane is not None:
        ba = ba * plane
        bc = bc * plane
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    if not (os.path.exists(NEUTRAL_PATH) and os.path.exists(GRASP_PATH)
            and os.path.exists(DANG_PATH)):
        print("THIẾU calib (neutral/grasp/dang) — chạy calib_all trước!")
        sys.exit(1)
    neutral = load_json(NEUTRAL_PATH)["neutral_pos"]
    grasp = load_json(GRASP_PATH)["grasp_pos"]
    dang = load_json(DANG_PATH)

    n1 = int(neutral.get("1", 0)); g1 = int(grasp.get("1", 0))
    n2 = int(neutral.get("2", 0)); g2 = int(grasp.get("2", 0))
    n4 = int(neutral.get("4", 0)); g4 = int(grasp.get("4", 0))
    n7 = int(neutral.get("7", 0)); g7 = int(grasp.get("7", 0))
    n6 = int(neutral.get("6", 0)); k6 = int(dang.get("k6_max", 0))
    n9 = int(neutral.get("9", 0)); k9 = int(dang.get("k9_max", 0))
    d5 = int(neutral.get("5", 0)); d8 = int(neutral.get("8", 0))

    gcfg = {}
    if os.path.exists(GRASP_CFG_PATH):
        gcfg = load_json(GRASP_CFG_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("LOI: khong mo duoc webcam")
        sys.exit(1)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1)
    ts0 = int(round(time.time() * 1000))

    portHandler = None
    pk = None
    try:
        portHandler, pk = serial_servo.connect()
        for s in GRASP_SERVOS:
            serial_servo.enable_torque(pk, s)
        serial_servo.move_servo(pk, 1, n1); serial_servo.move_servo(pk, 2, n2)
        serial_servo.move_servo(pk, 4, n4); serial_servo.move_servo(pk, 7, n7)
        serial_servo.move_servo(pk, 6, n6); serial_servo.move_servo(pk, 9, n9)
        serial_servo.move_servo(pk, 5, d5); serial_servo.move_servo(pk, 8, d8)
        time.sleep(0.5)

        fmax = [40.0, 30.0, 60.0, 60.0]  # tf, to, ft, fm — ƯỚC NGƯỜI (gập
        # hết tay: cái gập 40° — cái đối 30° — trỏ 60° — giữa 60°) — KHÔNG đo!
        ymax_khep = 10.0  # góc xích 2 ngón (khép hết) — set sát (yt thật 5-15°)!
        print(f"FMAX set tay = {fmax} | ymax_khep = {ymax_khep} (ước người!)")
        with vision.HandLandmarker.create_from_options(options) as landmarker:

            tf_max, to_max, ft_max, fm_max = fmax
            ema = [None] * 5
            calib = {"phase": "xoe", "start": time.time(),
                     "d_max": 0.0, "d_min": 1e9}
            th = {}
            if os.path.exists(LOAD_PATH):
                th = load_json(LOAD_PATH).get("thresholds", {})
            print("GRASP TELEOP — MỞ/KHÉP TAY (gập hết = CẦM!) — ESC = thoát")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(
                    mp_image, int(round(time.time() * 1000)) - ts0)
                if result.hand_landmarks:
                    lm = result.hand_landmarks[0]
                    pts = np.array([[l.x, l.y, l.z] for l in lm])
                    mid = (pts[5] + pts[9]) / 2
                    d = (np.hypot(mid[0] - pts[8][0], mid[1] - pts[8][1]) +
                         np.hypot(mid[0] - pts[12][0], mid[1] - pts[12][1]))
                    g = [180 - angle_between(pts[1], pts[2], pts[3]),
                         90 - angle_between(pts[0], pts[1], pts[4]),
                         180 - angle_between(pts[5], pts[6], pts[7]),
                         180 - angle_between(pts[9], pts[10], pts[11]), d]
                    for i in range(5):
                        if ema[i] is None:
                            ema[i] = g[i]
                        else:
                            ema[i] = ema[i] + (g[i] - ema[i]) / EMA_N
                    tf, to, ft, fm, d = ema
                    if calib["phase"] == "xoe":
                        calib["d_max"] = max(calib["d_max"], d)
                        state = f"CALIB XOE ({calib['d_max']:.3f}) - XOE HET 5s!"
                        if time.time() - calib["start"] > 5:
                            calib["phase"], calib["start"] = "chum", time.time()
                    elif calib["phase"] == "chum":
                        calib["d_min"] = min(calib["d_min"], d)
                        state = f"CALIB CHUM ({calib['d_min']:.3f}) - CHUM 5s!"
                        if time.time() - calib["start"] > 5:
                            calib["phase"] = "run"
                            print(f"CALIB XONG d_max={calib['d_max']:.3f} "
                                  f"d_min={calib['d_min']:.3f}")
                            gcfg["d_max"] = calib["d_max"]
                            gcfg["d_min"] = calib["d_min"]
                            with open(GRASP_CFG_PATH, "w", encoding="utf-8") as f:
                                json.dump(gcfg, f, ensure_ascii=False, indent=2)
                    if calib["phase"] != "run":
                        for s, p in zip(GRASP_SERVOS,
                                        [n1, n2, n4, n7, n6, n9, d5, d8]):
                            serial_servo.move_servo(pk, s, p)
                    else:
                        ratio = (d - calib["d_min"]) / (
                            calib["d_max"] - calib["d_min"] + 1e-9)
                        ratio = clamp(ratio, 0.0, 1.0)
                        p1 = int(n1 + (tf / tf_max) * (g1 - n1))
                        p2 = int(n2 + (to / to_max) * (g2 - n2))
                        p4 = int(n4 + (ft / ft_max) * (g4 - n4))
                        p7 = int(n7 + (fm / fm_max) * (g7 - n7))
                        p6 = int(n6 + ratio * (k6 - n6))
                        p9 = int(n9 + ratio * (k9 - n9))
                        p1 = clamp(p1, min(n1, g1), max(n1, g1))
                        p2 = clamp(p2, min(n2, g2), max(n2, g2))
                        p4 = clamp(p4, min(n4, g4), max(n4, g4))
                        p7 = clamp(p7, min(n7, g7), max(n7, g7))
                        p6 = clamp(p6, min(n6, k6), max(n6, k6))
                        p9 = clamp(p9, min(n9, k9), max(n9, k9))
                        serial_servo.move_servo(pk, 1, p1)
                        serial_servo.move_servo(pk, 2, p2)
                        serial_servo.move_servo(pk, 4, p4)
                        serial_servo.move_servo(pk, 7, p7)
                        serial_servo.move_servo(pk, 6, p6)
                        serial_servo.move_servo(pk, 9, p9)
                        serial_servo.move_servo(pk, 5, d5)
                        serial_servo.move_servo(pk, 8, d8)
                    if calib["phase"] == "run":
                        state = "MO"
                    if th:
                        loads = [serial_servo.read_load(pk, s) or 0 for s in LOAD_DETECT_SERVOS]
                        if any(loads[i] > th.get(str(s), 500) for i, s in enumerate(LOAD_DETECT_SERVOS)):
                            state = "CẦM!"
                    cv2.putText(frame, f"tf:{tf:5.1f} to:{to:5.1f} ft:{ft:5.1f} "
                                f"fm:{fm:5.1f} d:{d:.3f} | {state}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    if calib["phase"] == "run":
                        print(f"1:{p1} 2:{p2} 4:{p4} 7:{p7} 6:{p6} 9:{p9} | {state}",
                              flush=True)
                cv2.imshow("GRASP - ESC de thoat", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                time.sleep(SAMPLE_MS)
        print("THOÁT GRASP.")
    finally:
        if pk is not None and portHandler is not None:
            serial_servo.disconnect(portHandler)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
