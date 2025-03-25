import asyncio
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
import threading
import queue

from vllm.config import CacheConfig, ModelConfig, VllmConfig
from .decoder import tokens_decoder_sync


class OrpheusModel:
    def __init__(self, model_name, dtype=torch.bfloat16, gpu_memory_utilization: float = 0.9, cpu_offload_gb: float = 0):
        self.model_name = self._map_model_params(model_name)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.cpu_offload_gb = cpu_offload_gb
        self.dtype = dtype
        self.engine = self._setup_engine()
        self.available_voices = ["zoe", "zac",
                                 "jess", "leo", "mia", "julia", "leah"]
        self.tokeniser = AutoTokenizer.from_pretrained(self.model_name)

    def _map_model_params(self, model_name):
        model_map = {
            # "nano-150m":{
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            # },
            # "micro-400m":{
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            # },
            # "small-1b":{
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            # },
            "medium-3b": {
                "repo_id": "canopylabs/orpheus-3b-0.1-ft",
            },
        }
        unsupported_models = ["nano-150m", "micro-400m", "small-1b"]
        if (model_name in unsupported_models):
            raise ValueError(
                f"Model {model_name} is not supported. Only medium-3b is supported, small, micro and nano models will be released very soon")
        elif model_name in model_map:
            return model_map[model_name]["repo_id"]
        else:
            return model_name

    def _setup_engine(self):
        print(
            f"Setting up engine with model {self.model_name}, dtype {self.dtype}, gpu_memory_utilization {self.gpu_memory_utilization}, cpu_offload_gb {self.cpu_offload_gb}")
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            dtype=self.dtype,
            
        )
        engine_config = VllmConfig(
            model_config=ModelConfig(
                model_path=self.model_name,
                dtype=self.dtype,
                tokenizer=self.tokeniser,
                tokenizer_mode="auto",
                trust_remote_code=True,
            ),
            cache_config=CacheConfig(
                block_size=None,
                swap_space=4,
                cache_dtype="auto",
                gpu_memory_utilization=self.gpu_memory_utilization,
                cpu_offload_gb=self.cpu_offload_gb,
            )
        )
        return AsyncLLMEngine.from_engine_args(engine_args, engine_config)

    def validate_voice(self, voice):
        if voice:
            if voice not in self.engine.available_voices:
                raise ValueError(
                    f"Voice {voice} is not available for model {self.model_name}")

    def _format_prompt(self, prompt, voice="tara", model_type="larger"):
        if model_type == "smaller":
            if voice:
                return f"<custom_token_3>{prompt}[{voice}]<custom_token_4><custom_token_5>"
            else:
                return f"<custom_token_3>{prompt}<custom_token_4><custom_token_5>"
        else:
            if voice:
                adapted_prompt = f"{voice}: {prompt}"
                prompt_tokens = self.tokeniser(
                    adapted_prompt, return_tensors="pt")
                # start_token = torch.tensor([[128259]], dtype=torch.int64)
                start_token = torch.tensor([[128259]])
                end_tokens = torch.tensor(
                    # [[128009, 128260, 128261, 128257]], dtype=torch.int64)
                    [[128009, 128260, 128261, 128257]])
                all_input_ids = torch.cat(
                    [start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokeniser.decode(all_input_ids[0])
                return prompt_string
            else:
                prompt_tokens = self.tokeniser(prompt, return_tensors="pt")
                # start_token = torch.tensor([[128259]], dtype=torch.int64)
                start_token = torch.tensor([[128259]])
                end_tokens = torch.tensor(
                    # [[128009, 128260, 128261, 128257]], dtype=torch.int64)
                    [[128009, 128260, 128261, 128257]])
                all_input_ids = torch.cat(
                    [start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokeniser.decode(all_input_ids[0])
                return prompt_string

    def generate_tokens_sync(self, prompt, voice=None, request_id="req-001", temperature=0.6, top_p=0.8, max_tokens=1200, stop_token_ids=[49158], repetition_penalty=1.3):
        prompt_string = self._format_prompt(prompt, voice)
        print(prompt)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,  # Adjust max_tokens as needed.
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
        )

        token_queue = queue.Queue()

        async def async_producer():
            async for result in self.engine.generate(prompt=prompt_string, sampling_params=sampling_params, request_id=request_id):
                # Place each token text into the queue.
                token_queue.put(result.outputs[0].text)
            token_queue.put(None)  # Sentinel to indicate completion.

        def run_async():
            asyncio.run(async_producer())

        thread = threading.Thread(target=run_async)
        thread.start()

        while True:
            token = token_queue.get()
            if token is None:
                break
            yield token

        thread.join()

    def generate_speech(self, **kwargs):
        return tokens_decoder_sync(self.generate_tokens_sync(**kwargs))
