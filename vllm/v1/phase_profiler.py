# simple_timer.py
import time
import json
import os
from datetime import datetime
import csv


class SimpleTimer:
    """简单的计时器"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.records = []
            cls._instance.start_times = {}
        return cls._instance

    def start(self, stage_name):
        """开始计时"""
        self.start_times[stage_name] = time.perf_counter()
        print(f"[TIMER] ⏱️  Started: {stage_name}")

    def end(self, stage_name):
        """结束计时"""
        if stage_name not in self.start_times:
            print(f"[TIMER] Warning: {stage_name} not started")
            return

        # 秒 (s)
        elapsed = time.perf_counter() - self.start_times[stage_name]

        self.records.append({
            'stage': stage_name,
            'time_s': elapsed,
            'timestamp': datetime.now().isoformat()
        })

        print(f"[TIMER] Finished: {stage_name} ({elapsed:.4f} s)")
        del self.start_times[stage_name]

    def save(self, filepath='vllm_timing.json', csv_filepath=None):
        """
        保存到JSON和CSV文件
        """
        dir_name = os.path.dirname(filepath)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump({
                'records': self.records,
            }, f, indent=1)

        print(f"[TIMER] JSON saved to: {filepath}")

        if csv_filepath:
            self._save_csv(csv_filepath)

    def _save_csv(self, csv_filepath):
        """保存到CSV文件（追加或创建）"""

        dir_name = os.path.dirname(csv_filepath)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # stage : time_s
        row_data = {record['stage']: record['time_s'] for record in self.records}

        file_exists = os.path.exists(csv_filepath)

        if file_exists:
            with open(csv_filepath, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_fieldnames = reader.fieldnames

            all_fieldnames = list(existing_fieldnames)

            for stage in row_data.keys():
                if stage not in all_fieldnames:
                    all_fieldnames.append(stage)

            with open(csv_filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_fieldnames)
                writer.writerow(row_data)

        else:
            fieldnames = row_data.keys()

            with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row_data)


# 全局单例
timer = SimpleTimer()

# simple_timer.py
import time
import json
import os
from datetime import datetime
import csv


class SimpleTimer:
    """简单的计时器"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.records = []
            cls._instance.start_times = {}
        return cls._instance

    def start(self, stage_name):
        """开始计时"""
        self.start_times[stage_name] = time.perf_counter()
        print(f"[TIMER] ⏱️  Started: {stage_name}")

    def end(self, stage_name):
        """结束计时"""
        if stage_name not in self.start_times:
            print(f"[TIMER] Warning: {stage_name} not started")
            return

        # 秒 (s)
        elapsed = time.perf_counter() - self.start_times[stage_name]

        self.records.append({
            'stage': stage_name,
            'time_s': elapsed,
            'timestamp': datetime.now().isoformat()
        })

        print(f"[TIMER] Finished: {stage_name} ({elapsed:.4f} s)")
        del self.start_times[stage_name]

    def save(self, filepath='vllm_timing.json', csv_filepath=None):
        """
        保存到JSON和CSV文件
        """
        dir_name = os.path.dirname(filepath)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump({
                'records': self.records,
            }, f, indent=1)

        print(f"[TIMER] JSON saved to: {filepath}")

        if csv_filepath:
            self._save_csv(csv_filepath)

    def _save_csv(self, csv_filepath):
        """保存到CSV文件（追加或创建）"""

        dir_name = os.path.dirname(csv_filepath)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # stage : time_s
        row_data = {record['stage']: record['time_s'] for record in self.records}

        file_exists = os.path.exists(csv_filepath)

        if file_exists:
            with open(csv_filepath, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_fieldnames = reader.fieldnames

            all_fieldnames = list(existing_fieldnames)

            for stage in row_data.keys():
                if stage not in all_fieldnames:
                    all_fieldnames.append(stage)

            with open(csv_filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_fieldnames)
                writer.writerow(row_data)

        else:
            fieldnames = row_data.keys()

            with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row_data)


# 全局单例
timer = SimpleTimer()
