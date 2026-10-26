"""Optional local Hugging Face models. Importing this module downloads nothing.

Qwen2.5 text models follow the official Transformers chat-template interface;
Qwen2.5-VL uses the upstream Qwen2_5_VLForConditionalGeneration class.
These are intent controllers, not medically aligned segmentation models.
"""
from pathlib import Path

import torch


class LocalQwenGenerator:
    def __init__(self, model_path, *, vision=False, device="cpu", max_new_tokens=384):
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError("Provide a downloaded LOCAL Qwen model directory")
        if not 32 <= int(max_new_tokens) <= 2048:
            raise ValueError("max_new_tokens must be between 32 and 2048")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError("Install transformers in a separate compatible LLM environment; "
                              "no dependencies or weights are installed automatically") from exc
        self.vision, self.device = bool(vision), torch.device(device)
        self.max_new_tokens = int(max_new_tokens)
        dtype = torch.float32 if self.device.type == "cpu" else torch.float16
        kwargs = dict(local_files_only=True, trust_remote_code=False, torch_dtype=dtype)
        if self.vision:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(
                str(path), local_files_only=True, trust_remote_code=False,
                max_pixels=1280 * 28 * 28)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(path), **kwargs)
        else:
            self.processor = AutoTokenizer.from_pretrained(
                str(path), local_files_only=True, trust_remote_code=False)
            self.model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def generate(self, system, user, *, images=()):
        if images and not self.vision:
            raise ValueError("A text model cannot read MRI/overlay images")
        if len(images) > 6:
            raise ValueError("Select at most six bound ROI views")
        if self.vision:
            from PIL import Image
            # Local image paths only; do not let generated text trigger a network fetch.
            visuals = []
            for item in images:
                path = Path(item).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                with Image.open(path) as im:
                    visuals.append(im.convert("RGB").copy())
            content = [{"type": "image", "image": im} for im in visuals]
            content.append({"type": "text", "text": user})
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": content}]
            text = self.processor.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
            batch = self.processor(text=[text], images=visuals or None,
                                   padding=True, return_tensors="pt")
        else:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": user}]
            text = self.processor.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
            batch = self.processor([text], return_tensors="pt")
        batch = batch.to(self.device)
        tokens = self.model.generate(**batch, max_new_tokens=self.max_new_tokens,
                                     do_sample=False)
        tail = tokens[:, batch["input_ids"].shape[1]:]
        return self.processor.batch_decode(tail, skip_special_tokens=True)[0]
