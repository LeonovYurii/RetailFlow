from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class ResponseGenerator:
    def __init__(self, adapter_path: str | Path):
        if not torch.cuda.is_available():
            raise RuntimeError("This generator requires CUDA-enabled PyTorch.")

        adapter_path = str(adapter_path)
        config = PeftConfig.from_pretrained(adapter_path)

        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float32,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            quantization_config=quantization,
            device_map={"": 0},
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=False,
        )
        self.model.eval()
        self.model.config.use_cache = True

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 128,
    ) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda:0")

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = output[0, inputs["input_ids"].shape[1]:]

        return self.tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        ).strip()