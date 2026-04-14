"""
Logging and visualization utilities for minGPT training.
Supports console, file, JSON logging, TensorBoard, wandb, and web dashboard.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from collections import defaultdict
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from mingpt.utils import CfgNode as CN


# ----------------------------------------------------------------------------- 
# Structured JSON Logger

class JSONLogger:
    """Logs metrics to JSON Lines format for easy parsing."""
    
    def __init__(self, log_dir: str, filename: str = 'metrics.jsonl'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / filename
    
    def log(self, step: int, metrics: Dict[str, Any]):
        """Log metrics as a JSON line."""
        record = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
    
    def read_metrics(self) -> List[Dict[str, Any]]:
        """Read all logged metrics."""
        if not self.log_file.exists():
            return []
        metrics = []
        with open(self.log_file, 'r') as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
        return metrics


# -----------------------------------------------------------------------------
# Training Logger

class TrainingLogger:
    """
    Comprehensive training logger with multiple backends.
    
    Features:
    - Console logging with formatting
    - File logging
    - JSON structured logging
    - TensorBoard integration
    - Weights & Biases (wandb) integration
    - Training report generation
    """
    
    @staticmethod
    def get_default_config():
        C = CN()
        # log directory
        C.log_dir = './logs'
        # log to console
        C.console_log = True
        # log to file
        C.file_log = True
        # log to JSON
        C.json_log = True
        # TensorBoard logging
        C.tensorboard = False
        # wandb logging
        C.wandb = False
        # wandb project name
        C.wandb_project = 'mingpt'
        # wandb run name (None for auto-generated)
        C.wandb_run_name = None
        # log interval (steps)
        C.log_interval = 10
        # generate training report at the end
        C.generate_report = True
        return C
    
    def __init__(self, config, model: Optional[nn.Module] = None):
        self.config = config
        self.model = model
        
        # create log directory
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # setup loggers
        self._setup_console_logger()
        self._setup_file_logger()
        self.json_logger = JSONLogger(str(self.log_dir)) if config.json_log else None
        
        # TensorBoard
        self.tb_writer = None
        if config.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(str(self.log_dir / 'tensorboard'))
                self.console_logger.info("TensorBoard logging enabled")
            except ImportError:
                self.console_logger.warning("TensorBoard not available, install with: pip install tensorboard")
        
        # wandb
        self.wandb_run = None
        if config.wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    dir=str(self.log_dir),
                    config=config.to_dict() if hasattr(config, 'to_dict') else vars(config)
                )
                self.console_logger.info(f"wandb logging enabled: {wandb.run.url}")
            except ImportError:
                self.console_logger.warning("wandb not available, install with: pip install wandb")
        
        # metrics history
        self.metrics_history = defaultdict(list)
        self.start_time = time.time()
    
    def _setup_console_logger(self):
        """Setup console logger."""
        self.console_logger = logging.getLogger('mingpt.console')
        self.console_logger.setLevel(logging.INFO)
        
        # clear existing handlers
        self.console_logger.handlers = []
        
        if self.config.console_log:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.console_logger.addHandler(handler)
    
    def _setup_file_logger(self):
        """Setup file logger."""
        self.file_logger = logging.getLogger('mingpt.file')
        self.file_logger.setLevel(logging.INFO)
        
        # clear existing handlers
        self.file_logger.handlers = []
        
        if self.config.file_log:
            log_file = self.log_dir / 'training.log'
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.file_logger.addHandler(handler)
    
    def log(self, message: str, level: str = 'info'):
        """Log a message to all enabled backends."""
        if level == 'info':
            self.console_logger.info(message)
            self.file_logger.info(message)
        elif level == 'warning':
            self.console_logger.warning(message)
            self.file_logger.warning(message)
        elif level == 'error':
            self.console_logger.error(message)
            self.file_logger.error(message)
    
    def log_metrics(self, step: int, metrics: Dict[str, Union[float, int]], prefix: str = ''):
        """
        Log metrics to all enabled backends.
        
        Args:
            step: Current training step
            metrics: Dictionary of metric names to values
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        # store in history
        for key, value in metrics.items():
            full_key = f"{prefix}{key}" if prefix else key
            self.metrics_history[full_key].append((step, value))
        
        # console log (at intervals)
        if step % self.config.log_interval == 0:
            metric_str = ', '.join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                    for k, v in metrics.items()])
            self.log(f"Step {step}: {metric_str}")
        
        # JSON log
        if self.json_logger:
            self.json_logger.log(step, {f"{prefix}{k}": v for k, v in metrics.items()})
        
        # TensorBoard
        if self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f"{prefix}{key}", value, step)
        
        # wandb
        if self.wandb_run:
            log_dict = {f"{prefix}{k}": v for k, v in metrics.items()}
            log_dict['step'] = step
            self.wandb.log(log_dict)
    
    def log_histogram(self, step: int, tag: str, values: torch.Tensor):
        """Log histogram of values (for gradients, weights, etc.)."""
        if self.tb_writer:
            self.tb_writer.add_histogram(tag, values, step)
        
        if self.wandb_run:
            self.wandb.log({tag: self.wandb.Histogram(values.cpu().numpy()), 'step': step})
    
    def log_model_gradients(self, step: int):
        """Log gradient statistics for all model parameters."""
        if self.model is None:
            return
        
        grad_norms = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                
                # log per-layer gradient norms to TensorBoard
                if self.tb_writer and step % 100 == 0:  # less frequent for histograms
                    self.tb_writer.add_scalar(f'gradients/{name}_norm', grad_norm, step)
        
        if grad_norms:
            avg_grad_norm = sum(grad_norms) / len(grad_norms)
            max_grad_norm = max(grad_norms)
            self.log_metrics(step, {
                'grad_norm_avg': avg_grad_norm,
                'grad_norm_max': max_grad_norm
            }, prefix='train/')
    
    def log_learning_rate(self, step: int, lr: float):
        """Log learning rate."""
        self.log_metrics(step, {'learning_rate': lr}, prefix='train/')
    
    def log_text_sample(self, step: int, tag: str, text: str):
        """Log a text sample (e.g., generated text)."""
        if self.tb_writer:
            self.tb_writer.add_text(tag, text, step)
        
        if self.wandb_run:
            self.wandb.log({tag: self.wandb.Html(f"<pre>{text}</pre>"), 'step': step})
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a training report with summary statistics.
        
        Returns:
            Path to generated report
        """
        if output_path is None:
            output_path = self.log_dir / 'training_report.md'
        else:
            output_path = Path(output_path)
        
        # compute statistics
        total_time = time.time() - self.start_time
        
        report_lines = [
            "# minGPT Training Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Training Time:** {total_time / 3600:.2f} hours",
            "",
            "## Metrics Summary",
            "",
        ]
        
        # add metrics summary
        for metric_name, values in sorted(self.metrics_history.items()):
            if values:
                steps, vals = zip(*values)
                report_lines.extend([
                    f"### {metric_name}",
                    f"- **Final:** {vals[-1]:.6f}",
                    f"- **Best:** {min(vals):.6f}",
                    f"- **Mean:** {sum(vals) / len(vals):.6f}",
                    ""
                ])
        
        # write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.log(f"Training report saved to: {output_path}")
        return str(output_path)
    
    def close(self):
        """Close all loggers and generate final report."""
        if self.config.generate_report:
            self.generate_report()
        
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.wandb_run:
            self.wandb_run.finish()


# -----------------------------------------------------------------------------
# Web Dashboard (Simple HTTP Server)

class WebDashboard:
    """
    Simple web dashboard for monitoring training progress.
    Serves a basic HTML page with training metrics.
    """
    
    def __init__(self, log_dir: str, port: int = 8080):
        self.log_dir = Path(log_dir)
        self.port = port
        self.server = None
        self.thread = None
    
    def start(self):
        """Start the web dashboard in a background thread."""
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import threading
            
            log_dir = self.log_dir
            
            class DashboardHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    
                    # read metrics
                    metrics_file = log_dir / 'metrics.jsonl'
                    metrics = []
                    if metrics_file.exists():
                        with open(metrics_file, 'r') as f:
                            for line in f:
                                if line.strip():
                                    metrics.append(json.loads(line))
                    
                    # generate simple HTML
                    html = self._generate_html(metrics, log_dir)
                    self.wfile.write(html.encode())
                
                def _generate_html(self, metrics: List[Dict], log_dir: Path) -> str:
                    latest = metrics[-1] if metrics else {}
                    
                    html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>minGPT Training Dashboard</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            h1 {{ color: #333; }}
                            .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                            .metric-label {{ font-weight: bold; }}
                            pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
                        </style>
                        <meta http-equiv="refresh" content="30">
                    </head>
                    <body>
                        <h1>minGPT Training Dashboard</h1>
                        <p>Log directory: {log_dir}</p>
                        <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <h2>Latest Metrics</h2>
                    """
                    
                    for key, value in latest.items():
                        if key not in ['step', 'timestamp']:
                            html += f'<div class="metric"><span class="metric-label">{key}:</span> {value}</div>\n'
                    
                    html += f"""
                        <h2>Training Log</h2>
                        <pre>{json.dumps(metrics[-10:], indent=2)}</pre>
                    </body>
                    </html>
                    """
                    return html
                
                def log_message(self, format, *args):
                    pass  # suppress logging
            
            self.server = HTTPServer(('localhost', self.port), DashboardHandler)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            
            print(f"Web dashboard started at http://localhost:{self.port}")
            
        except Exception as e:
            print(f"Failed to start web dashboard: {e}")
    
    def stop(self):
        """Stop the web dashboard."""
        if self.server:
            self.server.shutdown()
            print("Web dashboard stopped")


# -----------------------------------------------------------------------------
# Utility Functions

def get_logger(config, model: Optional[nn.Module] = None) -> TrainingLogger:
    """Factory function to create a TrainingLogger."""
    return TrainingLogger(config, model)


def start_dashboard(log_dir: str, port: int = 8080) -> WebDashboard:
    """Start a web dashboard for monitoring."""
    dashboard = WebDashboard(log_dir, port)
    dashboard.start()
    return dashboard
