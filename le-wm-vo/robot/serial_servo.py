"""
Serial servo control for bionic hand 8-DOF via SC Servo SDK.
Uses scservo_sdk (SCS CL protocol) on COM13 at 1Mbps.
"""
import sys
import os
import json
import time

# SDK path
sys.path.append(r"d:\ai_training\tools\STServo_Python\STServo_Python\stservo-env")

try:
    from scservo_sdk import PortHandler, scscl
    HAS_SDK = True
except ImportError:
    HAS_SDK = False

SERVO_IDS = [1, 2, 4, 5, 6, 7, 8, 9]
PORT = "COM13"
BAUDRATE = 1000000
TORQUE_REG = 40


def connect(port=PORT, baudrate=BAUDRATE):
    """Open connection and enable torque on all servos."""
    if not HAS_SDK:
        raise RuntimeError("scservo_sdk not installed. Check venv path.")
    
    portHandler = PortHandler(port)
    packetHandler = scscl(portHandler)
    
    if not portHandler.openPort():
        raise RuntimeError(f"Failed to open {port}")
    if not portHandler.setBaudRate(baudrate):
        raise RuntimeError(f"Failed to set baudrate {baudrate}")
    
    # Enable torque
    for sid in SERVO_IDS:
        packetHandler.write1ByteTxRx(sid, TORQUE_REG, 1)
    
    # Stabilize: đợi USB-UART + servo nội bộ ổn định sau torque enable
    time.sleep(0.3)
    
    return portHandler, packetHandler


def move_servo(packetHandler, servo_id, position, speed=0, accel=0):
    """Move single servo to position (0-1023)."""
    packetHandler.WritePos(servo_id, int(position), speed, accel)


def move_all(packetHandler, positions):
    """Move servos.
    Args:
        positions: dict {servo_id: position} — moves only specified servos.
    """
    if isinstance(positions, dict):
        for sid_str, pos in positions.items():
            sid = int(sid_str)
            move_servo(packetHandler, sid, pos)
    else:
        for sid, pos in zip(SERVO_IDS, positions):
            move_servo(packetHandler, sid, pos)


def move_from_json(packetHandler, json_path):
    """Load position from JSON file and move all servos.
    JSON format: {"grasp_pos": {"1": 550, "2": 650, ...}} or {"neutral_pos": {...}}
    """
    with open(json_path) as f:
        data = json.load(f)
    
    # Find the position key
    pos_key = [k for k in data if "_pos" in k][0]
    positions = data[pos_key]
    
    print(f"Moving to {pos_key}...")
    print(f"  Servo values: {{{', '.join(f'{k}:{v}' for k, v in positions.items())}}}")
    move_all(packetHandler, positions)
    time.sleep(0.5)


def disconnect(portHandler):
    """Disable torque and close port."""
    packetHandler = scscl(portHandler)
    for sid in SERVO_IDS:
        packetHandler.write1ByteTxRx(sid, TORQUE_REG, 0)
    portHandler.closePort()
    print("Servos disabled, port closed.")


def release_torque(packetHandler, servo_id):
    """Release torque — user can rotate servo freely."""
    return packetHandler.write1ByteTxRx(servo_id, TORQUE_REG, 0)


def enable_torque(packetHandler, servo_id):
    """Enable torque — servo holds position."""
    return packetHandler.write1ByteTxRx(servo_id, TORQUE_REG, 1)


def read_position(packetHandler, servo_id):
    """Read current servo position (0-1023). Returns int or None."""
    pos, result, error = packetHandler.ReadPos(servo_id)
    return pos if result == 0 else None


def read_position(packetHandler, servo_id):
    """Read actual servo position. Returns int or None."""
    pos, speed, result, error = packetHandler.ReadPosSpeed(servo_id)
    if result == 0:
        return pos
    return None


def read_current(packetHandler, servo_id):
    """Read servo current draw (mA). Returns int or None.
    SCServo memory address 69-70 = PRESENT_CURRENT."""
    current, result, error = packetHandler.read2ByteTxRx(servo_id, 69)
    if result == 0:
        return packetHandler.scs_tohost(current, 15)
    return None


def read_load(packetHandler, servo_id):
    """Read servo motor load. Higher = more resistance/force.
    SC09 address 60-61 = PRESENT_LOAD (idle=0, moving=1000-2000, blocked=high).
    """
    load, result, error = packetHandler.read2ByteTxRx(servo_id, 60)
    if result == 0:
        return load  # raw value, no conversion needed
    return None


def read_all_loads(packetHandler):
    """Read load for all 8 servos. Returns dict {sid: load}."""
    loads = {}
    for sid in SERVO_IDS:
        l = read_load(packetHandler, sid)
        loads[sid] = l if l is not None else 0
    return loads


def is_grasping(packetHandler, threshold=500):
    """Detect grasp via motor load. Returns (True, sid, load) or (False, None, None)."""
    for sid in SERVO_IDS:
        l = read_load(packetHandler, sid)
        if l is not None and l > threshold:
            return True, sid, l
    return False, None, None


if __name__ == "__main__":
    if not HAS_SDK:
        print("[LỖI] scservo_sdk không tìm thấy. Kiểm tra venv path.")
        sys.exit(1)
    
    if len(sys.argv) >= 3 and sys.argv[1] == "--move":
        json_path = sys.argv[2]
        port = sys.argv[3] if len(sys.argv) > 3 else PORT
        
        # Auto-resolve data/calib/ path
        if not os.path.exists(json_path):
            alt = os.path.join(os.path.dirname(__file__), "..", "data/calib", os.path.basename(json_path))
            if os.path.exists(alt):
                json_path = alt
                print(f"  Resolved to: {os.path.normpath(json_path)}")
        
        print(f"Connecting {port}...")
        portHandler, packetHandler = connect(port=port)
        move_from_json(packetHandler, json_path)
        disconnect(portHandler)
        print("✓ Done")
    else:
        # Quick test: connect, enable, move servo 1 slightly, disable
        print(f"Connecting {PORT}...")
        portHandler, packetHandler = connect()
        print(f"Moving servo 1 to 500...")
        move_servo(packetHandler, 1, 500)
        time.sleep(0.5)
        print(f"Moving servo 1 back to 800...")
        move_servo(packetHandler, 1, 800)
        time.sleep(0.5)
        disconnect(portHandler)
        print("✓ Test done")
