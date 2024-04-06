"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Tuple, Optional
import random

# import torch
# from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams


def run_vllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
) -> float:
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        # block_size=16,
        # swap_space=100,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=0.96,
    )

    # Add the requests to the engine.
    for prompt, _, output_len in requests:
        sampling_params = SamplingParams(
            n=n,
            temperature=0.0 if use_beam_search else 1.0,
            top_p=1.0,
            use_beam_search=use_beam_search,
            ignore_eos=True,
            max_tokens=output_len,
            logprobs=4
        )
        # print(output_len)
        # FIXME(woosuk): Do not use internal method.
        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )

    start = time.perf_counter()
    # FIXME(woosuk): Do not use internal method.

    # Include the decoding time.
    result = llm._run_engine(use_tqdm=True)
    input_num_tokens = []
    output_num_tokens = []

    for res in result:
        input_num_tokens.append(len(res.prompt_token_ids))
        output_num_tokens.append(
            len(res.outputs[0].token_ids) + len(res.prompt_token_ids))

    end = time.perf_counter()
    return end - start, input_num_tokens, output_num_tokens


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    if args.dataset is None:
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len)
                    for _ in range(args.num_samples)]

    else:
        with open(args.dataset) as f:
            requests = json.load(f)

    if args.num_samples is not None:
        requests = requests[0:args.num_samples]

    elapsed_time, input_num_tokens, output_num_tokens = run_vllm(
        requests, args.model, args.tokenizer, args.quantization, args.tensor_parallel_size, args.seed, 1, False, args.trust_remote_code, args.dtype, None, False)
    prompt_num_tokens = sum(prompt_len for prompt_len in input_num_tokens)
    total_num_tokens = sum(output_len for output_len in output_num_tokens)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s \n"
          f"Tokens/s: {total_num_tokens / elapsed_time:.2f} tokens/s \n"
          f"Prompt_num_tokens:{prompt_num_tokens:.2f} tokens \n"
          f"Total_num_tokens:{total_num_tokens:.2f} tokens \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to the dataset.")
    parser.add_argument("--model", type=str, default="meta/llama2-70b")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--input-len", type=int, default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor Parallel Size")
    parser.add_argument("--output-len", type=int, default=None,
                        help="Output length for each request")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of first few samples used for inference test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--quantization', type=str,
                        default=None, help="Quantization Method")
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None
        assert args.output_len is None

    main(args)
