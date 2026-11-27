#!/usr/bin/env python3
"""Pretrained SAM3 domain adaptation; compatible with torchrun DDP."""
import importlib
import foundation_prompt

if __name__ == '__main__':
    print(importlib.import_module(foundation_prompt.name+'.foundation_prompt_v2.sam_domain_pipeline').main())
