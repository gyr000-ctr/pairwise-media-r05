import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import gridcc  # noqa: E402


def pytest_collection_modifyitems(items):
    """兼容垫片：既有 tests/test_core.py 以裸名 GridCC 引用公开类
    （该文件不可改动且未显式 import 该名称）。在收集后将包的公开 API
    注入对应测试模块的全局命名空间，仅作用于本包测试模块。"""
    public = {
        name: getattr(gridcc, name)
        for name in ("GridCC", "GridShapeError", "PixelValueError", "BACKGROUND")
    }
    seen = set()
    for item in items:
        module = getattr(item, "module", None)
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        if getattr(module, "__name__", "").startswith("tests.") or \
                "test_core" in getattr(module, "__name__", ""):
            for name, value in public.items():
                module.__dict__.setdefault(name, value)
