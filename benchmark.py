import time
import torch
import matplotlib.pyplot as plt

class Benchmark:
    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self):
        self.start_time = None
        self.prev_time = None
        self.token_times = []
        self.tokens = 0
        self.vram_allocated = []
        self.vram_reserved = []
        self.vram_peak = 0

    def _record_vram(self):
        if self.device == 'cuda':
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)   # MB
            reserv = torch.cuda.memory_reserved() / (1024 ** 2)   # MB
            self.vram_allocated.append(alloc)
            self.vram_reserved.append(reserv)
            self.vram_peak = max(self.vram_peak, alloc)

    def start(self):
        if self.device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.start_time = time.perf_counter()
        self.prev_time = self.start_time
        self.token_times = []
        self.tokens = 0
        self.vram_allocated = []
        self.vram_reserved = []
        self.vram_peak = 0
        self._record_vram()

    def step(self):
        if self.device == 'cuda':
            torch.cuda.synchronize()

        now = time.perf_counter()
        duration = now - self.prev_time
        self.token_times.append(duration)

        self.prev_time = now
        self.tokens += 1
        self._record_vram()

    def stop(self):
        if self.device == 'cuda':
            torch.cuda.synchronize()

        total_time = time.perf_counter() - self.start_time

        if len(self.token_times) > 1:
            first_token_ms = self.token_times[0] * 1000
            rest_tokens_ms = self.token_times[1:]
            avg_ms = (sum(rest_tokens_ms) / len(rest_tokens_ms)) * 1000
            min_ms = min(rest_tokens_ms) * 1000
            max_ms = max(rest_tokens_ms) * 1000
        elif len(self.token_times) == 1:
            first_token_ms = avg_ms = min_ms = max_ms = self.token_times[0] * 1000
        else:
            first_token_ms = avg_ms = min_ms = max_ms = 0

        tok_per_s = self.tokens / total_time if total_time > 0 else 0

        print("\n" + "="*30)
        print("📊 BENCHMARK RESULT")
        print("="*30)
        print(f"Total Time       : {total_time:.4f} s")
        print(f"Tokens Gen       : {self.tokens}")
        print(f"Throughput       : {tok_per_s:.2f} tok/s")
        print(f"First Token Lat. : {first_token_ms:.2f} ms")
        print(f"Next Token Avg   : {avg_ms:.2f} ms")
        print(f"Next Token Min   : {min_ms:.2f} ms")
        print(f"Next Token Max   : {max_ms:.2f} ms")

        if self.device == 'cuda':
            print(f"Peak VRAM Alloc  : {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
            print(f"Peak VRAM Reserv : {torch.cuda.max_memory_reserved() / (1024**2):.2f} MB")

        print("="*30)

        self.plot()

    def plot(self):
        if len(self.token_times) == 0:
            print("No data to plot.")
            return

        token_ms = [t * 1000 for t in self.token_times]

        window = 5
        moving_avg = []
        for i in range(len(token_ms)):
            start = max(0, i - window + 1)
            avg = sum(token_ms[start:i+1]) / (i - start + 1)
            moving_avg.append(avg)

        plt.figure(figsize=(10, 5))
        plt.plot(token_ms, label="Token Latency (ms)")
        plt.plot(moving_avg, linestyle='--', label="Moving Avg")

        if self.device == 'cuda' and len(self.vram_allocated) > 1:
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax2.plot(self.vram_allocated, color='red', alpha=0.6, label="VRAM Allocated (MB)")
            ax2.plot(self.vram_reserved, color='orange', alpha=0.6, linestyle=':', label="VRAM Reserved (MB)")
            ax2.set_ylabel("VRAM (MB)")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        else:
            plt.legend()

        plt.xlabel("Token Index")
        plt.ylabel("Latency (ms)")
        plt.title("LLM Token Latency Benchmark")
        plt.tight_layout()
        plt.show()
        plt.close()