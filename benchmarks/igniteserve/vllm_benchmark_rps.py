import json
import asyncio
import aiohttp
import time
import numpy as np
import argparse
import csv
from typing import List, Optional
from dataclasses import dataclass
from collections import Counter
import os

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
    send_timestamp: float = 0.0  # 新增：记录请求发出的时间
    attempts: int = 1


class ShareGPTBenchmark:
    def __init__(self, base_url, model, dataset_path, num_prompts,
                 max_workers, output_csv, request_rate, hotspot_key=None,
                 seed=0, max_tokens=256, max_retries=0,
                 request_timeout=600.0):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.dataset_path = dataset_path
        self.num_prompts = num_prompts
        self.max_workers = max_workers
        self.output_csv = output_csv
        self.request_rate = request_rate  # 新增：请求到达率 (req/s)
        self.hotspot_key = hotspot_key
        self.rng = np.random.default_rng(seed)
        self.max_tokens = max_tokens
        self.max_retries = max(0, max_retries)
        self.request_timeout = request_timeout
        self.dataset = self._load_dataset()

    # def _load_dataset(self) -> List[str]:
    #     with open(self.dataset_path, 'r', encoding='utf-8') as f:
    #         data = json.load(f)
    #     prompts = []
    #     for item in data:
    #         conversations = item.get("conversations", [])
    #         human_msg = next((c.get("value", "").strip() for c in conversations if c.get("from") == "human"), None)
    #         if human_msg and len(human_msg) >= 10:
    #             prompts.append(human_msg[:1000])
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
        send_timestamp = time.time()
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
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
                    return RequestResult(req_id, False, time.time() - send_timestamp,
                                         error=f"HTTP {resp.status}: {err[:100]}", send_timestamp=send_timestamp)

                async for line in resp.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: ') and '[DONE]' not in line_str:
                        # 【核心修复】：拦截服务端的异常传递
                        data_str = line_str[6:]
                        try:
                            data_json = json.loads(data_str)
                            if isinstance(data_json, dict) and "error" in data_json:
                                return RequestResult(req_id, False, time.time() - send_timestamp,
                                                     error=f"Server Error: {data_json['error']}",
                                                     send_timestamp=send_timestamp)
                        except json.JSONDecodeError:
                            pass
                        out_tokens += 1
                        current_time = time.time()
                        if first_token_time is None:
                            first_token_time = current_time
                            ttft = first_token_time - send_timestamp

                latency = time.time() - send_timestamp
                tpot = (latency - ttft) / (out_tokens - 1) if out_tokens > 1 and ttft else None

                if out_tokens == 0:
                    return RequestResult(req_id, False, latency, error="0 Output Tokens generated",
                                         send_timestamp=send_timestamp)

                return RequestResult(req_id, True, latency, ttft=ttft, tpot=tpot, input_tokens=len(prompt) // 3,
                                     output_tokens=out_tokens, send_timestamp=send_timestamp)
        except Exception as e:
            return RequestResult(req_id, False, time.time() - send_timestamp, error=repr(e),
                                 send_timestamp=send_timestamp)

    async def run(self):
        mode_str = f"{self.request_rate} req/s (Poisson)" if self.request_rate else "尽力而为 (As fast as possible)"
        print(f"启动压测: 发送 {self.num_prompts} 个请求 | 注入模式: {mode_str}")

        start_time = time.time()
        results = []
        sem = asyncio.Semaphore(self.max_workers)

        async def task_wrapper(session, prompt, req_id):
            """限制并发的请求包装器"""
            async with sem:
                res = await self._send_request(session, prompt, req_id)
                attempts = 1
                retryable_errors = (
                    "TimeoutError(", "ServerDisconnectedError(",
                    "ClientOSError(", "ClientPayloadError(",
                )
                while (
                    not res.success
                    and attempts <= self.max_retries
                    and (res.error or "").startswith(retryable_errors)
                    and res.output_tokens == 0
                ):
                    attempts += 1
                    await asyncio.sleep(0.1)
                    res = await self._send_request(session, prompt, req_id)
                res.attempts = attempts
                results.append(res)

        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            for i, prompt in enumerate(self.dataset):
                # 【核心逻辑】：泊松到达的时间间隔模拟
                if self.request_rate and self.request_rate > 0:
                    # 指数分布公式：计算下一个请求需要等待的时间
                    delay = self.rng.exponential(1.0 / self.request_rate)
                    await asyncio.sleep(delay)

                tasks.append(asyncio.create_task(task_wrapper(session, prompt, i)))

            # 等待所有分发出去的任务执行完毕
            await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        self._print_stats(results, total_time)
        self._save_to_csv(results)

    def _print_stats(self, results, total_time):

        success = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print("\n" + "=" * 50)
        print(f"总耗时: {total_time:.2f}s | 成功: {len(success)} | 失败: {len(failed)}")
        retried = sum(r.attempts > 1 for r in results)
        print(f"连接级重试请求数: {retried}")

        if failed:
            print("\n【失败原因归类】")
            for err, cnt in Counter(r.error for r in failed).items():
                print(f" - {err}: {cnt}次")

        if success:
            # 提取数据
            latencies = [r.latency for r in success if hasattr(r, 'latency') and r.latency]
            ttfts = [r.ttft for r in success if r.ttft]
            tpots = [r.tpot for r in success if r.tpot]
            total_out = sum(r.output_tokens for r in success)

            # 计算所需指标（增加判空容错）
            avg_latency = np.mean(latencies) if latencies else 0.0
            avg_ttft = np.mean(ttfts) if ttfts else 0.0
            p99_ttft = np.percentile(ttfts, 99) if ttfts else 0.0
            avg_tpot = np.mean(tpots) if tpots else 0.0
            p99_tpot = np.percentile(tpots, 99) if tpots else 0.0
            throughput = total_out / total_time if total_time > 0 else 0.0

            # 终端打印输出
            print("\n【延迟统计】")
            print(f" 平均时延:   {avg_latency:.4f} 秒")
            print(f" 平均 TTFT:  {avg_ttft:.4f} 秒")
            print(f" P90 TTFT:   {np.percentile(ttfts, 90):.4f} 秒" if ttfts else " P90 TTFT: N/A")
            print(f" P99 TTFT:   {p99_ttft:.4f} 秒")
            print(f" 平均 TPOT:  {avg_tpot:.4f} 秒")
            print(f" P90 TPOT:   {np.percentile(tpots, 90):.4f} 秒" if tpots else " P90 TPOT: N/A")
            print(f" P99 TPOT:   {p99_tpot:.4f} 秒")
            print(f" 总输出吞吐: {throughput:.2f} tokens/s")

            # ==========================================
            # 新增：将宏观统计指标导出到单独的 summary CSV
            # ==========================================
            if hasattr(self, 'output_csv') and self.output_csv:
                try:
                    # 生成摘要文件名 (例如: data.csv -> data_summary.csv)
                    base_name, ext = os.path.splitext(self.output_csv)
                    summary_csv_path = f"{base_name}_summary{ext}"

                    # 确保目录存在
                    dir_name = os.path.dirname(summary_csv_path)
                    if dir_name:
                        os.makedirs(dir_name, exist_ok=True)

                    # 写入汇总数据 (采用追加模式 'a'，方便你把多个 RPS 的结果写进同一个文件；
                    # 如果你想每次覆盖，可以将 mode='a' 改为 mode='w')
                    file_exists = os.path.exists(summary_csv_path)
                    with open(summary_csv_path, mode='w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)

                        # 写入表头
                        writer.writerow([
                            "total_time_s",
                            "avg_latency_s",
                            "throughput_tokens_per_s",
                            "avg_ttft_s",
                            "p99_ttft_s",
                            "avg_tpot_s",
                            "p99_tpot_s"
                        ])

                        # 写入具体数据
                        writer.writerow([
                            f"{total_time:.4f}",
                            f"{avg_latency:.4f}",
                            f"{throughput:.4f}",
                            f"{avg_ttft:.4f}",
                            f"{p99_ttft:.4f}",
                            f"{avg_tpot:.4f}",
                            f"{p99_tpot:.4f}"
                        ])
                    print(f"\n[+] 宏观摘要数据已成功保存至: {summary_csv_path}")
                except Exception as e:
                    print(f"\n[-] 保存 Summary CSV 失败: {str(e)}")

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
                    "ttft_s", "tpot_s", "input_tokens", "output_tokens",
                    "attempts", "error_msg"
                ])

                # 为了后续画图分析，按发送时间排序
                if hasattr(results[0], 'send_timestamp'):
                    results.sort(key=lambda x: x.send_timestamp)

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
                        r.attempts,
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
    parser.add_argument('--max-workers', type=int, default=100, help="限制最大并发数 (在使用 rate 时可适当调大)")
    parser.add_argument('--output-csv', type=str, default='benchmark_detailed_results.csv')
    parser.add_argument('--request-rate', type=float, default=None,
                        help="每秒请求到达率 (例如 2.0 表示每秒2个。若不指定，则全速发送)")
    parser.add_argument('--hotspot-key', type=str, default=None,
                        help="CUDA graph hotspot case key passed via X-VLLM-HOTSPOT-KEY")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed for reproducible Poisson arrivals")
    parser.add_argument('--max-tokens', type=int, default=256,
                        help="Maximum output tokens per request")
    parser.add_argument('--max-retries', type=int, default=0,
                        help="Retries for zero-token connection failures")
    parser.add_argument('--request-timeout', type=float, default=600.0,
                        help="Per-request timeout in seconds")
    args = parser.parse_args()

    benchmark = ShareGPTBenchmark(
        base_url=args.base_url,
        model=args.model,
        dataset_path=args.dataset,
        num_prompts=args.num_prompts,
        max_workers=args.max_workers,
        output_csv=args.output_csv,
        request_rate=args.request_rate,
        hotspot_key=args.hotspot_key,
        seed=args.seed,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
    )
    asyncio.run(benchmark.run())
