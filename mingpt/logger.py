"""
Training visualization and logging for minGPT.
Supports console logging, file logging, JSON structured logs,
TensorBoard integration, and optional wandb support.
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, asdict

import torch
import numpy as np

from mingpt.utils import CfgNode as CN


@dataclass
class LogEntry:
    timestamp: str
    iter_num: int
    metrics: Dict[str, float]
    extra: Dict[str, Any]


class TrainingLogger:
    """
    Comprehensive training logger with multiple output backends:
    - Console output with progress display
    - File logging for persistent records
    - JSON structured logs for programmatic access
    - TensorBoard integration for visualization
    - Optional wandb integration
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.log_dir = 'logs'
        C.experiment_name = 'minGPT'
        C.use_tensorboard = True
        C.use_wandb = False
        C.wandb_project = 'minGPT'
        C.wandb_entity = None
        C.log_interval = 100
        C.save_json_logs = True
        C.console_level = logging.INFO
        C.file_level = logging.DEBUG
        return C

    def __init__(self, config, model_config=None):
        self.config = config
        self.model_config = model_config
        
        self.log_dir = Path(config.log_dir) / config.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = time.time()
        self.iter_num = 0
        self.metrics_history: List[LogEntry] = []
        
        self._setup_console_logger()
        self._setup_file_logger()
        
        self.tensorboard_writer = None
        if config.use_tensorboard:
            self._setup_tensorboard()
        
        self.wandb_run = None
        if config.use_wandb:
            self._setup_wandb()

    def _setup_console_logger(self):
        self.console_logger = logging.getLogger('minGPT.console')
        self.console_logger.setLevel(self.config.console_level)
        self.console_logger.handlers = []
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.config.console_level)
        formatter = logging.Formatter('%(asctime)s | %(message)s', 
                                      datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
        self.console_logger.addHandler(console_handler)

    def _setup_file_logger(self):
        self.file_logger = logging.getLogger('minGPT.file')
        self.file_logger.setLevel(self.config.file_level)
        self.file_logger.handlers = []
        
        log_file = self.log_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.config.file_level)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        self.file_logger.addHandler(file_handler)

    def _setup_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.log_dir / 'tensorboard'
            self.tensorboard_writer = SummaryWriter(log_dir=str(tb_dir))
            self.console_logger.info(f"TensorBoard logging enabled: {tb_dir}")
        except ImportError:
            self.console_logger.warning(
                "TensorBoard not available. Install with: pip install tensorboard"
            )
            self.tensorboard_writer = None

    def _setup_wandb(self):
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.experiment_name,
                config=self.model_config.to_dict() if self.model_config else {},
                dir=str(self.log_dir),
            )
            self.console_logger.info(f"wandb logging enabled: {wandb.run.url}")
        except ImportError:
            self.console_logger.warning(
                "wandb not available. Install with: pip install wandb"
            )
            self.wandb_run = None
        except Exception as e:
            self.console_logger.warning(f"wandb initialization failed: {e}")
            self.wandb_run = None

    def log(self, iter_num: int, metrics: Dict[str, float], 
            extra: Optional[Dict[str, Any]] = None):
        self.iter_num = iter_num
        
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            iter_num=iter_num,
            metrics=metrics,
            extra=extra or {},
        )
        self.metrics_history.append(entry)
        
        self._log_to_console(iter_num, metrics)
        self._log_to_file(iter_num, metrics, extra)
        
        if self.tensorboard_writer:
            self._log_to_tensorboard(iter_num, metrics, extra)
        
        if self.wandb_run:
            self._log_to_wandb(iter_num, metrics, extra)
        
        if self.config.save_json_logs:
            self._save_json_entry(entry)

    def _log_to_console(self, iter_num: int, metrics: Dict[str, float]):
        elapsed = time.time() - self.start_time
        metrics_str = ' | '.join(f'{k}: {v:.4f}' for k, v in metrics.items())
        msg = f'iter {iter_num} | time {elapsed:.1f}s | {metrics_str}'
        self.console_logger.info(msg)

    def _log_to_file(self, iter_num: int, metrics: Dict[str, float],
                     extra: Optional[Dict[str, Any]]):
        elapsed = time.time() - self.start_time
        metrics_str = ' | '.join(f'{k}: {v:.6f}' for k, v in metrics.items())
        msg = f'iter {iter_num} | elapsed {elapsed:.2f}s | {metrics_str}'
        if extra:
            extra_str = ' | '.join(f'{k}: {v}' for k, v in extra.items())
            msg += f' | {extra_str}'
        self.file_logger.info(msg)

    def _log_to_tensorboard(self, iter_num: int, metrics: Dict[str, float],
                           extra: Optional[Dict[str, Any]]):
        for name, value in metrics.items():
            self.tensorboard_writer.add_scalar(f'metrics/{name}', value, iter_num)
        
        if extra:
            if 'learning_rate' in extra:
                self.tensorboard_writer.add_scalar(
                    'optimizer/learning_rate', extra['learning_rate'], iter_num
                )
            if 'grad_norm' in extra:
                self.tensorboard_writer.add_scalar(
                    'gradients/norm', extra['grad_norm'], iter_num
                )

    def _log_to_wandb(self, iter_num: int, metrics: Dict[str, float],
                     extra: Optional[Dict[str, Any]]):
        import wandb
        log_dict = {'iter': iter_num, **metrics}
        if extra:
            log_dict.update(extra)
        wandb.log(log_dict, step=iter_num)

    def _save_json_entry(self, entry: LogEntry):
        json_file = self.log_dir / 'metrics.jsonl'
        with open(json_file, 'a') as f:
            f.write(json.dumps(asdict(entry)) + '\n')

    def log_histogram(self, name: str, values: torch.Tensor, iter_num: int):
        if self.tensorboard_writer:
            self.tensorboard_writer.add_histogram(
                name, values.detach().cpu().numpy(), iter_num
            )
        
        if self.wandb_run:
            import wandb
            wandb.log({name: wandb.Histogram(values.detach().cpu().numpy())}, 
                     step=iter_num)

    def log_gradients(self, model: torch.nn.Module, iter_num: int):
        if not self.tensorboard_writer and not self.wandb_run:
            return
        
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_norm += grad_norm ** 2
                
                if self.tensorboard_writer:
                    self.tensorboard_writer.add_scalar(
                        f'gradients/{name}', grad_norm, iter_num
                    )
        
        total_norm = total_norm ** 0.5
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(
                'gradients/total_norm', total_norm, iter_num
            )
        
        return total_norm

    def log_text_sample(self, name: str, text: str, iter_num: int):
        if self.tensorboard_writer:
            self.tensorboard_writer.add_text(name, text, iter_num)
        
        if self.wandb_run:
            import wandb
            wandb.log({name: wandb.Html(f'<pre>{text}</pre>')}, step=iter_num)

    def log_model_summary(self, model: torch.nn.Module):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
        }
        
        self.console_logger.info(
            f"Model: {trainable_params:,} trainable params, "
            f"{total_params - trainable_params:,} non-trainable params"
        )
        
        if self.wandb_run:
            import wandb
            wandb.config.update(summary)
        
        return summary

    def generate_report(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        
        report = {
            'experiment_name': self.config.experiment_name,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_time_seconds': elapsed,
            'total_iterations': self.iter_num,
            'final_metrics': self.metrics_history[-1].metrics if self.metrics_history else {},
            'best_metrics': self._get_best_metrics(),
            'iterations_per_second': self.iter_num / elapsed if elapsed > 0 else 0,
        }
        
        report_file = self.log_dir / 'training_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self._generate_text_report(report)
        
        return report

    def _get_best_metrics(self) -> Dict[str, Dict[str, Any]]:
        best_metrics = {}
        if not self.metrics_history:
            return best_metrics
        
        metric_names = self.metrics_history[0].metrics.keys()
        
        for name in metric_names:
            values = [(e.iter_num, e.metrics[name]) for e in self.metrics_history]
            best_min = min(values, key=lambda x: x[1])
            best_max = max(values, key=lambda x: x[1])
            
            best_metrics[name] = {
                'min_value': best_min[1],
                'min_at_iter': best_min[0],
                'max_value': best_max[1],
                'max_at_iter': best_max[0],
            }
        
        return best_metrics

    def _generate_text_report(self, report: Dict[str, Any]):
        report_file = self.log_dir / 'training_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"Training Report: {report['experiment_name']}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Start Time: {report['start_time']}\n")
            f.write(f"End Time: {report['end_time']}\n")
            f.write(f"Total Time: {report['total_time_seconds']:.2f} seconds\n")
            f.write(f"Total Iterations: {report['total_iterations']}\n")
            f.write(f"Iterations/Second: {report['iterations_per_second']:.2f}\n\n")
            
            f.write("Final Metrics:\n")
            for name, value in report['final_metrics'].items():
                f.write(f"  {name}: {value:.6f}\n")
            f.write("\n")
            
            f.write("Best Metrics:\n")
            for name, stats in report['best_metrics'].items():
                f.write(f"  {name}:\n")
                f.write(f"    Min: {stats['min_value']:.6f} (iter {stats['min_at_iter']})\n")
                f.write(f"    Max: {stats['max_value']:.6f} (iter {stats['max_at_iter']})\n")

    def close(self):
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.wandb_run:
            import wandb
            wandb.finish()
        
        self.generate_report()


class WebMonitor:
    """
    Simple web-based monitoring dashboard for training progress.
    Provides real-time visualization of training metrics.
    """

    def __init__(self, log_dir: str, port: int = 6007):
        self.log_dir = Path(log_dir)
        self.port = port
        self.server = None

    def start(self):
        try:
            from http.server import HTTPServer, SimpleHTTPRequestHandler
            import threading
            
            class MonitorHandler(SimpleHTTPRequestHandler):
                def __init__(self, *args, log_dir=None, **kwargs):
                    self.log_dir = log_dir
                    super().__init__(*args, directory=str(log_dir), **kwargs)
            
            handler = lambda *args, **kwargs: MonitorHandler(*args, log_dir=self.log_dir, **kwargs)
            self.server = HTTPServer(('localhost', self.port), handler)
            
            server_thread = threading.Thread(target=self.server.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            
            print(f"Web monitor started at http://localhost:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to start web monitor: {e}")
            return False

    def stop(self):
        if self.server:
            self.server.shutdown()
            print("Web monitor stopped")


def generate_dashboard_html(log_dir: str, output_path: Optional[str] = None):
    log_path = Path(log_dir)
    metrics_file = log_path / 'metrics.jsonl'
    
    if not metrics_file.exists():
        print(f"No metrics file found at {metrics_file}")
        return
    
    metrics_data = []
    with open(metrics_file, 'r') as f:
        for line in f:
            metrics_data.append(json.loads(line))
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>minGPT Training Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .chart-container { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 30px; }
        .stats { display: flex; flex-wrap: wrap; gap: 20px; }
        .stat-card { background: white; padding: 15px; border-radius: 8px; min-width: 200px; }
        .stat-value { font-size: 24px; font-weight: bold; color: #2196F3; }
        .stat-label { color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>minGPT Training Monitor</h1>
"""
    
    if metrics_data:
        latest = metrics_data[-1]
        html_content += "<div class='stats'>"
        for name, value in latest['metrics'].items():
            html_content += f"""
                <div class='stat-card'>
                    <div class='stat-value'>{value:.4f}</div>
                    <div class='stat-label'>{name}</div>
                </div>
            """
        html_content += "</div>"
        
        metric_names = list(metrics_data[0]['metrics'].keys())
        
        for metric_name in metric_names:
            html_content += f"""
        <div class='chart-container'>
            <h2>{metric_name}</h2>
            <canvas id='chart_{metric_name}'></canvas>
        </div>
"""
    
    html_content += """
    </div>
    <script>
"""
    
    if metrics_data:
        iterations = [d['iter_num'] for d in metrics_data]
        
        for metric_name in metric_names:
            values = [d['metrics'][metric_name] for d in metrics_data]
            html_content += f"""
        var ctx_{metric_name} = document.getElementById('chart_{metric_name}').getContext('2d');
        var chart_{metric_name} = new Chart(ctx_{metric_name}, {{
            type: 'line',
            data: {{
                labels: {iterations},
                datasets: [{{
                    label: '{metric_name}',
                    data: {values},
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1,
                    fill: false
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{ title: {{ display: true, text: 'Iteration' }} }},
                    y: {{ title: {{ display: true, text: '{metric_name}' }} }}
                }}
            }}
        }});
"""
    
    html_content += """
    </script>
</body>
</html>
"""
    
    output_file = output_path or str(log_path / 'monitor.html')
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Dashboard generated at {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Logging and monitoring tools for minGPT')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    dashboard_parser = subparsers.add_parser('dashboard', help='Generate monitoring dashboard')
    dashboard_parser.add_argument('--log-dir', type=str, required=True,
                                 help='Log directory containing metrics.jsonl')
    dashboard_parser.add_argument('--output', type=str,
                                 help='Output HTML file path')
    
    monitor_parser = subparsers.add_parser('monitor', help='Start web monitor')
    monitor_parser.add_argument('--log-dir', type=str, required=True,
                               help='Log directory to serve')
    monitor_parser.add_argument('--port', type=int, default=6007,
                               help='Port for web server')
    
    args = parser.parse_args()
    
    if args.command == 'dashboard':
        generate_dashboard_html(args.log_dir, args.output)
    elif args.command == 'monitor':
        monitor = WebMonitor(args.log_dir, args.port)
        monitor.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
