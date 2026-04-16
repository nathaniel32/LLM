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

    def start(self):
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        self.prev_time = self.start_time
        self.token_times = []
        self.tokens = 0

    def step(self):
        """Call tepat setelah token baru dihasilkan"""
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        now = time.perf_counter()
        # Hitung durasi sejak token sebelumnya atau sejak start
        duration = now - self.prev_time
        self.token_times.append(duration)
        
        self.prev_time = now
        self.tokens += 1

    def stop(self):
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        total_time = time.perf_counter() - self.start_time
        
        # Pisahkan First Token vs sisanya untuk analisis lebih dalam
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
        print(f'Total Time       : {total_time:.4f} s')
        print(f'Tokens Gen       : {self.tokens}')
        print(f'Throughput       : {tok_per_s:.2f} tok/s')
        print(f'First Token Lat. : {first_token_ms:.2f} ms') # Penting!
        print(f'Next Token Avg   : {avg_ms:.2f} ms')      # Lebih akurat untuk cache
        print(f'Next Token Min   : {min_ms:.2f} ms')
        print(f'Next Token Max   : {max_ms:.2f} ms')
        print("="*30)

        self.plot()

    def plot(self):
        if len(self.token_times) == 0:
            print("No data to plot.")
            return

        # Convert ke ms
        token_ms = [t * 1000 for t in self.token_times]

        # Moving average
        window = 5
        moving_avg = []
        for i in range(len(token_ms)):
            start = max(0, i - window + 1)
            avg = sum(token_ms[start:i+1]) / (i - start + 1)
            moving_avg.append(avg)

        # Plot
        plt.figure()
        plt.plot(token_ms, label="Token Latency (ms)")
        plt.plot(moving_avg, linestyle='--', label="Moving Avg")

        plt.xlabel("Token Index")
        plt.ylabel("Latency (ms)")
        plt.title("LLM Token Latency Benchmark")
        plt.legend()

        # Simpan ke file (penting buat skripsi)
        #plt.savefig("benchmark_latency.png", dpi=300)
        plt.show()