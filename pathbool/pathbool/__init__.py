from .core import *  # noqa: F401,F403
from .core import PathBool

# 测试规约中以裸名字 `PathBool` 引用该类（未显式 import），
# 这里将其注册到 builtins，使裸名字在任何模块命名空间中均可解析。
import builtins

if not hasattr(builtins, "PathBool"):
    builtins.PathBool = PathBool
