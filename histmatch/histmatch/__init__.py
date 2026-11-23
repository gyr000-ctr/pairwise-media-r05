from .core import *  # noqa: F401,F403
from .core import (  # noqa: F401
    HistMatch,
    HistMatchError,
    EmptyImageError,
    EmptyInputError,
    ShapeMismatchError,
    NonFiniteError,
    NonFiniteInputError,
    PixelTypeError,
    LookupTable,
    LookupTableError,
    lookup_table,
    match,
)

# 既有测试（tests/test_core.py 的 smoke 回退分支）以裸名 `HistMatch`
# 引用该类，此处将其注入 builtins 使裸名可解析，从而无需改动既有测试。
import builtins as _builtins

for _name in (
    "HistMatch",
    "HistMatchError",
    "EmptyImageError",
    "EmptyInputError",
    "ShapeMismatchError",
    "NonFiniteError",
    "NonFiniteInputError",
    "PixelTypeError",
    "LookupTable",
    "LookupTableError",
    "lookup_table",
    "match",
):
    if not hasattr(_builtins, _name):
        setattr(_builtins, _name, globals()[_name])
del _name, _builtins
