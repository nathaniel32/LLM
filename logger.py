import os
import json

class Logger:
    def __init__(self, out_dir, filename="metrics.json"):
        self.out_dir = out_dir
        self.log_path = os.path.join(out_dir, filename)
        os.makedirs(out_dir, exist_ok=True)
        self.data: dict = {}

    def set_meta(self, meta_dict: dict):            
        self.data.update(meta_dict)
        self._save()

    def log(self, category: str, metrics: dict):
        print(category, metrics)

        metrics_copy = metrics.copy()

        if category not in self.data:
            self.data[category] = []

        self.data[category].append(metrics_copy)
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
    logger.log("log", {"loss":3.4, "val-loss":3.2})
    logger.log("log", {"loss":1.2, "val-loss":1.2})
    logger.delete("model")