from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms import BaseLLM
from langchain.schema import Generation, LLMResult
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

# 与你原脚本保持一致的模型名称与量化配置
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


class Qwen25LLM(BaseLLM):
    """将本地 Qwen2.5-7B-Instruct 封装为 LangChain LLM，用于评论 Agent。"""

    tokenizer: Any = None
    model: Any = None
    generation_config: Any = None

    def __init__(self, temperature: float = 0.2, max_new_tokens: int = 512):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            padding_side="left",
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=0.95,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Qwen, a helpful Chinese analyst assistant. "
                    "You follow the ReAct format strictly when using tools."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                generation_config=self.generation_config,
            )

        input_ids_len = model_inputs.input_ids.shape[1]
        generated_ids_slice = generated_ids[0][input_ids_len:]
        response = self.tokenizer.decode(
            generated_ids_slice,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        if stop:
            for stop_word in stop:
                if stop_word in response:
                    response = response.split(stop_word)[0]
        return response

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations: List[List[Generation]] = []
        for p in prompts:
            text = self._call(p, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": MODEL_NAME,
            "quantization": "4-bit NF4",
        }

    @property
    def _llm_type(self) -> str:
        return "qwen2.5-7b-instruct-local"

