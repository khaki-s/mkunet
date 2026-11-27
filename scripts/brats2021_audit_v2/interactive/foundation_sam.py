#!/usr/bin/env python3
"""Load the private project adapter without shadowing installed upstream nnUNet."""
import importlib
import foundation_prompt

if __name__=='__main__':
    print(importlib.import_module(foundation_prompt.name+'.foundation_prompt_v2.sam_fusion_pipeline').main())
