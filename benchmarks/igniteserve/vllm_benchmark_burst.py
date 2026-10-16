import json
import asyncio
import aiohttp
import time
import numpy as np
import argparse
import csv
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import Counter


@dataclass
class RequestResult:
    request_id: int
    success: bool
    latency: float
    ttft: Optional[float] = None
    tpot: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None


class ShareGPTBenchmark:
    def __init__(self, base_url, model, dataset_path, num_prompts, max_workers, output_csv, hotspot_key=None):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.dataset_path = dataset_path
        self.num_prompts = num_prompts
        self.max_workers = max_workers
        self.output_csv = output_csv
        self.hotspot_key = hotspot_key
        self.dataset = self._load_dataset()

    # def _load_dataset(self) -> List[str]:
    #     with open(self.dataset_path, 'r', encoding='utf-8') as f:
    #         data = json.load(f)
    #     prompts = []
    #     for item in data:
    #         conversations = item.get("conversations", [])
    #         human_msg = next((c.get("value", "").strip() for c in conversations if c.get("from") == "human"), None)
    #         if human_msg and len(human_msg) >= 10:
    #             prompts.append(human_msg[:1000])  # 限制输入长度防爆显存
    #         if len(prompts) >= self.num_prompts: break
    #     return prompts

    def _load_dataset(self) -> List[str]:
        import re
        prompts = []
        try:
            print(f"正在加载数据集: {self.dataset_path}")
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                conversations = item.get("conversations", [])
                human_msg = next((c.get("value", "").strip() for c in conversations if c.get("from") == "human"), None)
                if human_msg and len(human_msg) >= 10:
                    prompts.append(human_msg[:1000])
                if len(prompts) >= self.num_prompts: break
            return prompts

        except json.JSONDecodeError as e:
            print(f"\n[-] 警告: 标准 JSON 解析失败 ({e})。数据集文件可能下载不完整或已损坏。")
            print("[+] 正在启动容错提取模式，尝试强行从文件头部提取有效 Prompt...\n")

            # 备用方案：暴力读取文件前 100 万个字符，用正则提取
            with open(self.dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000000)  # 100万字符足够提取几百个Prompt了

                # 正则匹配 "from": "human", "value": "具体的提示词"
                pattern = r'"from":\s*"human",\s*"value":\s*"((?:\\.|[^"\\])*)"'
                matches = re.finditer(pattern, content)

                for match in matches:
                    # 简单处理转义字符
                    msg = match.group(1).replace('\\n', '\n').replace('\\"', '"')
                    if len(msg.strip()) >= 10:
                        prompts.append(msg.strip()[:1000])
                    if len(prompts) >= self.num_prompts:
                        break

            print(f"[+] 容错模式提取成功: 获取了 {len(prompts)} 个 Prompt。")
            return prompts

    async def _send_request(self, session, prompt, req_id):
        start_time = time.time()
        url = f"{self.base_url}/chat/completions"  # 强制使用 chat 接口
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.0,
            "stream": True
        }

        ttft, first_token_time = None, None
        out_tokens = 0

        headers = {}
        if self.hotspot_key:
            headers["X-VLLM-HOTSPOT-KEY"] = self.hotspot_key

        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    return RequestResult(req_id, False, time.time() - start_time,
                                         error=f"HTTP {resp.status}: {err[:100]}")

                async for line in resp.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: ') and '[DONE]' not in line_str:
                        # 只要有正常 data 流返回，就算一个 token
                        out_tokens += 1
                        current_time = time.time()
                        if first_token_time is None:
                            first_token_time = current_time
                            ttft = first_token_time - start_time

                latency = time.time() - start_time
                tpot = (latency - ttft) / (out_tokens - 1) if out_tokens > 1 and ttft else None

                if out_tokens == 0:
                    return RequestResult(req_id, False, latency,
                                         error="0 Output Tokens generated (Model rejected prompt)")

                return RequestResult(req_id, True, latency, ttft=ttft, tpot=tpot, input_tokens=len(prompt) // 3,
                                     output_tokens=out_tokens)
        except Exception as e:
            return RequestResult(req_id, False, time.time() - start_time, error=str(e))

    async def run(self):
        print(f"启动压测: {self.num_prompts} 个并发请求...")
        start_time = time.time()
        results = []
        sem = asyncio.Semaphore(self.max_workers)

        async def bounded_req(session, prompt, req_id):
            async with sem:
                return await self._send_request(session, prompt, req_id)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
            tasks = [bounded_req(session, p, i) for i, p in enumerate(self.dataset)]
            for task in asyncio.as_completed(tasks):
                results.append(await task)

        total_time = time.time() - start_time

        # 打印统计并保存 CSV
        self._print_stats(results, total_time)
        self._save_to_csv(results)

    def _print_stats(self, results, total_time):
        success = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print("\n" + "=" * 50)
        print(f"总耗时: {total_time:.2f}s | 成功: {len(success)} | 失败: {len(failed)}")

        if failed:
            print("\n【失败原因归类】")
            for err, cnt in Counter(r.error for r in failed).items():
                print(f" - {err}: {cnt}次")

        if success:
            ttfts = [r.ttft for r in success if r.ttft]
            tpots = [r.tpot for r in success if r.tpot]
            total_out = sum(r.output_tokens for r in success)

            print("\n【延迟统计 (冷启动看这里)】")
            print(f" 平均 TTFT:  {np.mean(ttfts):.4f} 秒")
            print(f" P90 TTFT:  {np.percentile(ttfts, 90):.4f} 秒")
            print(f" 平均 TPOT:  {np.mean(tpots):.4f} 秒")
            print(f" 总输出吞吐: {total_out / total_time:.2f} tokens/s")
        print("=" * 50)

    def _save_to_csv(self, results: List[RequestResult]):
        """将单个请求的详细数据保存到 CSV 文件中"""
        if not self.output_csv:
            return

        import os  # 确保导入了 os 模块

        try:
            # 1. 提取目标路径的目录部分
            dir_name = os.path.dirname(self.output_csv)
            # 2. 如果目录名不为空，且该目录不存在，则递归创建它
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
                print(f"[+] 检测到输出目录不存在，已自动创建: {dir_name}")
            # 3. 正常写入文件
            with open(self.output_csv, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow([
                    "request_id", "send_timestamp", "success", "latency_s",
                    "ttft_s", "tpot_s", "input_tokens", "output_tokens", "error_msg"
                ])
                # 写入数据
                for r in results:
                    writer.writerow([
                        r.request_id,
                        f"{r.send_timestamp:.4f}" if hasattr(r, 'send_timestamp') else "",
                        r.success,
                        f"{r.latency:.4f}",
                        f"{r.ttft:.4f}" if r.ttft is not None else "",
                        f"{r.tpot:.4f}" if r.tpot is not None else "",
                        r.input_tokens,
                        r.output_tokens,
                        r.error if r.error else ""
                    ])
            print(f"\n[+] 详细请求数据已成功保存至: {self.output_csv}")
        except Exception as e:
            print(f"\n[-] 保存 CSV 文件失败: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--num-prompts', type=int, default=100)
    parser.add_argument('--max-workers', type=int, default=10, help="限制冷启动瞬时并发")
    parser.add_argument('--output-csv', type=str, default='benchmark_detailed_results.csv',
                        help="指定输出的 CSV 文件路径")
    parser.add_argument('--hotspot-key', type=str, default=None,
                        help="CUDA graph hotspot case key passed via X-VLLM-HOTSPOT-KEY")
    args = parser.parse_args()

    benchmark = ShareGPTBenchmark(
        base_url=args.base_url,
        model=args.model,
        dataset_path=args.dataset,
        num_prompts=args.num_prompts,
        max_workers=args.max_workers,
        output_csv=args.output_csv,
        hotspot_key=args.hotspot_key
    )
    asyncio.run(benchmark.run())