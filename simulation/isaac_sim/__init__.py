"""Isaac Sim simulation for Go2 robot."""
# Isaac Sim interface requires Isaac Sim installation
# See: https://github.com/Zhefan-Xu/isaac-go2-ros2
# See: https://github.com/abizovnuralem/go2_omniverse

try:
    from .isaac_interface import IsaacInterface
    __all__ = ["IsaacInterface"]
except ImportError:
    __all__ = []

