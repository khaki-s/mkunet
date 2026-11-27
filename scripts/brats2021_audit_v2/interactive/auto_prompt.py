#!/usr/bin/env python3
"""Deprecated automatic-prompt entry; intentionally cannot train or infer."""
import sys
if __name__ == "__main__":
    print('旧 autoprompt_v1（随机初始化小 UNet 自动提示）已归档停用。其 val80/test100 结果下降；不会再次启动训练或测试。新方案入口：/data02/chujingyi/nnUNet/scripts/brats2021_audit_v2/interactive/foundation_prompt.py。历史代码与结果说明：/data02/chujingyi/brats_dataset_audit/foundation_autoprompt_20261113/archive/README.md', file=sys.stderr)
    raise SystemExit(2)
