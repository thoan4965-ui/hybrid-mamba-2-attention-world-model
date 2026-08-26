"""
Wrapper nhỏ dùng thư viện ncps (Neural Circuit Policies) chính thức
của Mathias Lechner — tác giả gốc của CfC.

Thay vì tự implement CfC từ đầu, ta dùng:
  pip install ncps

API chính:
  from ncps.torch import CfC
  from ncps.wirings import FullyConnected, AutoNCP

Không cần import file này nữa — module.py import trực tiếp từ ncps.
File này chỉ giữ lại để tham khảo.
"""
# File này đã được thay thế bởi ncps library.
# Xem module.py để biết cách sử dụng.
raise ImportError(
    "cfc.py đã bị deprecated. "
    "Dùng: from ncps.torch import CfC; from ncps.wirings import FullyConnected"
)
