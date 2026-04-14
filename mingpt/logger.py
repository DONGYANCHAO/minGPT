"""
Training logger with console, file, JSON logging, TensorBoard/WandB integration,
and simple web dashboard.
"""

import os
import json
import logging
import time
import webbrowser
from datetime import datetime
from typing import Dict, Optional, Any
from threading import Thread

import torch
import numpy as np

from mingpt.utils import CfgNode as CN

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Training Dashboard</title>
    <meta charset="UTF-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; padding: 20px; border-bottom: 2px solid #16213e; margin-bottom: 30px; }
        .header h1 { color: #e94560; font-size: 2.5em; }
        .card { background: #16213e; border-radius: 12px; padding: 25px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .card h2 { color: #0f3460; margin-bottom: 20px; font-size: 1.5em; border-bottom: 1px solid #0f3460; padding-bottom: 10px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .metric { background: linear-gradient(135deg, #0f3460, #533483); padding: 20px; border-radius: 8px; text-align: center; }
        .metric .label { font-size: 0.9em; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
        .metric .value { font-size: 2.5em; font-weight: bold; color: #e94560; margin-top: 10px; }
        .progress-bar { width: 100%; height: 20px; background: #0f3460; border-radius: 10px; overflow: hidden; margin: 20px 0; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #e94560, #ff6b6b); transition: width 0.3s ease; }
        .log-container { background: #0f0f23; padding: 20px; border-radius: 8px; max-height: 400px; overflow-y: auto; font-family: 'Consolas', monospace; font-size: 0.9em; }
        .log-line { padding: 5px 0; border-bottom: 1px solid #1a1a2e; }
        .log-info { color: #4ecdc4; }
        .log-warning { color: #ffe66d; }
        .log-error { color: #ff6b6b; }
        .refresh-note { text-align: center; color: #666; margin-top: 20px; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 minGPT Training Dashboard</h1>
            <p>Run: {{run_name}} | Started: {{start_time}}</p>
        </div>
        
        <div class="card">
            <h2>📊 Training Progress</h2>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {{progress}}%"></div>
            </div>
            <p style="text-align: center; font-size: 1.2em;">Iteration: {{current_iter}} / {{max_iter}} ({{progress:.1f}}%)</p>
        </div>
        
        <div class="card">
            <h2>📈 Current Metrics</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="label">Train Loss</div>
                    <div class="value">{{loss:.4f}}</div>
                </div>
                <div class="metric">
                    <div class="label">Learning Rate</div>
                    <div class="value">{{lr:.2e}}</div>
                </div>
                <div class="metric">
                    <div class="label">Iter/sec</div>
                    <div class="value">{{iters_per_sec:.2f}}</div>
                </div>
                <div class="metric">
                    <div class="label">ETA</div>
                    <div class="value">{{eta}}</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>📝 Training Log</h2>
            <div class="log-container">
                {{log_lines}}
            </div>
        </div>
        
        <div class="refresh-note">Refresh page for updates | Auto-refresh every 30s</div>
    </div>
    <script>setTimeout(() => location.reload(), 30000);</script>
</body>
</html>
"""

class TrainingLogger:

    @staticmethod
    def get_default_config():
        C = CN()
        C.enabled = True
        C.log_dir = './logs'
        C.console = True
        C.file = True
        C.json = True
        C.tensorboard = False
        C.wandb = False
        C.wandb_project = 'mingpt'
        C.web_dashboard = True
        C.log_interval = 10
        C.gradients = False
        C.parameters = False
        return C

    def __init__(self, config, run_name: Optional[str] = None):
        self.config = config
        self.run_name = run_name or f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.run_dir = os.path.join(config.log_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.metrics_history = []
        self.start_time = time.time()
        self.max_iters = 0
        self._setup_logging()
        self._setup_tensorboard()
        self._setup_wandb()

    def _setup_logging(self):
        self.logger = logging.getLogger('mingpt')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        if self.config.console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        if self.config.file:
            log_path = os.path.join(self.run_dir, 'training.log')
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.json_log_path = os.path.join(self.run_dir, 'metrics.json') if self.config.json else None
        self.dashboard_path = os.path.join(self.run_dir, 'dashboard.html')

    def _setup_tensorboard(self):
        self.tb_writer = None
        if self.config.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(os.path.join(self.run_dir, 'tensorboard'))
                self.logger.info("TensorBoard writer initialized")
            except ImportError:
                self.logger.warning("TensorBoard not available, install with: pip install tensorboard")

    def _setup_wandb(self):
        self.wandb = None
        if self.config.wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb.init(
                    project=self.config.wandb_project,
                    name=self.run_name,
                    dir=self.run_dir
                )
                self.logger.info("Weights & Biases initialized")
            except ImportError:
                self.logger.warning("W&B not available, install with: pip install wandb")

    def set_max_iters(self, max_iters: int):
        self.max_iters = max_iters

    def log_metrics(self, trainer, step: Optional[int] = None):
        if not self.config.enabled:
            return

        step = step if step is not None else trainer.iter_num
        metrics = {
            'step': step,
            'loss': trainer.loss.item() if trainer.loss is not None else 0,
            'lr': trainer.optimizer.param_groups[0]['lr'] if trainer.optimizer else 0,
            'iter_time': trainer.iter_dt,
            'iters_per_sec': 1.0 / trainer.iter_dt if trainer.iter_dt > 0 else 0,
            'timestamp': time.time()
        }

        if self.config.gradients and trainer.model:
            total_norm = 0.0
            for p in trainer.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            metrics['grad_norm'] = total_norm ** 0.5

        self.metrics_history.append(metrics)

        if self.json_log_path:
            with open(self.json_log_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)

        if self.tb_writer:
            for k, v in metrics.items():
                if k not in ['step', 'timestamp']:
                    self.tb_writer.add_scalar(k, v, step)

        if self.wandb:
            self.wandb.log(metrics, step=step)

        if self.config.web_dashboard:
            self._update_dashboard()

    def log_samples(self, samples: list, step: int, tag: str = 'samples'):
        if not self.config.enabled:
            return

        if self.tb_writer:
            for i, sample in enumerate(samples[:5]):
                self.tb_writer.add_text(f'{tag}/sample_{i}', str(sample), step)

        if self.wandb:
            self.wandb.log({tag: [self.wandb.Html(f"<pre>{s}</pre>") for s in samples[:10]]}, step=step)

    def _update_dashboard(self):
        if not self.metrics_history:
            return

        latest = self.metrics_history[-1]
        elapsed = time.time() - self.start_time
        current_iter = latest['step']
        
        progress = (current_iter / self.max_iters * 100) if self.max_iters > 0 else 0
        
        remaining = (self.max_iters - current_iter) * latest['iter_time'] if latest['iter_time'] > 0 else 0
        eta = time.strftime('%H:%M:%S', time.gmtime(remaining))

        log_lines = []
        for m in self.metrics_history[-50:]:
            loss = m['loss']
            line = f"[{m['step']}] loss={loss:.4f} lr={m['lr']:.2e}"
            log_lines.append(f'<div class="log-line log-info">{line}</div>')

        dashboard_html = DASHBOARD_TEMPLATE
        dashboard_html = dashboard_html.replace('{{run_name}}', self.run_name)
        dashboard_html = dashboard_html.replace('{{start_time}}', datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S'))
        dashboard_html = dashboard_html.replace('{{current_iter}}', str(current_iter))
        dashboard_html = dashboard_html.replace('{{max_iter}}', str(self.max_iters))
        dashboard_html = dashboard_html.replace('{{progress:.1f}}', f'{progress:.1f}')
        dashboard_html = dashboard_html.replace('{{progress}}', str(progress))
        dashboard_html = dashboard_html.replace('{{loss:.4f}}', f"{latest['loss']:.4f}")
        dashboard_html = dashboard_html.replace('{{lr:.2e}}', f"{latest['lr']:.2e}")
        dashboard_html = dashboard_html.replace('{{iters_per_sec:.2f}}', f"{latest['iters_per_sec']:.2f}")
        dashboard_html = dashboard_html.replace('{{eta}}', eta)
        dashboard_html = dashboard_html.replace('{{log_lines}}', ''.join(log_lines))

        with open(self.dashboard_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def open_dashboard(self):
        if os.path.exists(self.dashboard_path):
            webbrowser.open('file://' + os.path.abspath(self.dashboard_path))

    def final_report(self):
        import pandas as pd
        
        df = pd.DataFrame(self.metrics_history)
        report = {
            'run_name': self.run_name,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration': time.time() - self.start_time,
            'total_iterations': len(self.metrics_history),
            'final_loss': float(df['loss'].iloc[-1]),
            'best_loss': float(df['loss'].min()),
            'mean_loss': float(df['loss'].mean()),
        }

        report_path = os.path.join(self.run_dir, 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"\n{'='*50}")
        self.logger.info("TRAINING COMPLETE")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Run: {self.run_name}")
        self.logger.info(f"Duration: {report['duration']:.1f}s")
        self.logger.info(f"Total iterations: {report['total_iterations']}")
        self.logger.info(f"Final loss: {report['final_loss']:.4f}")
        self.logger.info(f"Best loss: {report['best_loss']:.4f}")
        self.logger.info(f"Report saved to: {report_path}")
        self.logger.info(f"{'='*50}")

        if self.wandb:
            self.wandb.finish()
        if self.tb_writer:
            self.tb_writer.close()

        return report

    def get_callback(self):
        def callback(trainer):
            if trainer.iter_num % self.config.log_interval == 0:
                self.log_metrics(trainer)
        return callback
