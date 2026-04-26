import os
import json

class Logger:
    def __init__(self, out_dir, filename="metrics.json"):
        self.out_dir = out_dir
        self.log_path = os.path.join(out_dir, filename)
        os.makedirs(out_dir, exist_ok=True)
        self.data: dict = {}
        self._load()

    def _load(self):
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

    def set_meta(self, meta_dict: dict):            
        self.data.update(meta_dict)
        self._save()

    def _sort(self, category, key):
        self.data[category] = sorted(self.data[category], key=lambda x: x.get(key, float('inf')))

    def log(self, category: str, metrics: dict, key: str = None):
        metrics_copy = metrics.copy()

        if category not in self.data:
            self.data[category] = []

        if key is not None and key in metrics_copy:
            for item in self.data[category]:
                if item.get(key) == metrics_copy[key]:
                    item.update(metrics_copy)
                    self._sort(category, key)
                    self._save()
                    return

        self.data[category].append(metrics_copy)
        self._sort(category, key)
        self._save()
        
    def _save(self):
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2)

    def delete(self, key: str):
        if key in self.data:
            del self.data[key]
            self._save()
            print(f"Key '{key}' was successfully deleted.")
        else:
            print(f"Key '{key}' not found.")

if __name__ == "__main__":
    logger = Logger("out/test")
    logger.set_meta({"param":300, "model":"gpt2-small"})
    logger.log("log", {"loss":3.4, "val-loss":3.2, "iter": 1}, "iter")
    logger.log("log", {"loss":1.2, "val-loss":1.2, "iter": 2}, "iter")
    logger.log("log", {"loss":1, "val-loss":1, "iter": 1}, "iter")
    logger.delete("model")