# corr_stats.py

import torch
import numpy as np



# 全局收集器
class CorrelationStatsCollector:
    """收集多层、多 batch 的相关性统计"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.corr_data = []  # 存储 (layer_idx, corr_tensor)
        self.corr_remote_data = []  # 只存远程部分

    def add(self, layer_idx: int, corr: torch.Tensor, corr_remote: torch.Tensor = None):
        """添加一次相关性数据"""
        self.corr_data.append({
            'layer': layer_idx,
            'corr': corr.detach().cpu(),  # [H, N]
        })
        if corr_remote is not None:
            self.corr_remote_data.append({
                'layer': layer_idx,
                'corr_remote': corr_remote.detach().cpu(),
            })

    def compute_stats(self):
        """计算统计量"""
        if not self.corr_data:
            print("No data collected!")
            return

        all_corr = torch.cat([d['corr'].flatten() for d in self.corr_data])
        all_corr_remote = torch.cat(
            [d['corr_remote'].flatten() for d in self.corr_remote_data]) if self.corr_remote_data else None

        print("=" * 60)
        print("Correlation Statistics (All)")
        print("=" * 60)
        print(f"  count: {all_corr.numel()}")
        print(f"  mean:  {all_corr.mean():.6f}")
        print(f"  std:   {all_corr.std():.6f}")
        print(f"  min:   {all_corr.min():.6f}")
        print(f"  max:   {all_corr.max():.6f}")

        quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
        q_values = torch.quantile(all_corr, torch.tensor(quantiles))
        print(f"  quantiles:")
        for q, v in zip(quantiles, q_values):
            print(f"    {q * 100:.0f}%: {v:.6f}")

        if all_corr_remote is not None:
            print("\n" + "=" * 60)
            print("Correlation Statistics (Remote Only)")
            print("=" * 60)
            print(f"  count: {all_corr_remote.numel()}")
            print(f"  mean:  {all_corr_remote.mean():.6f}")
            print(f"  std:   {all_corr_remote.std():.6f}")
            q_values_remote = torch.quantile(all_corr_remote, torch.tensor(quantiles))
            print(f"  quantiles:")
            for q, v in zip(quantiles, q_values_remote):
                print(f"    {q * 100:.0f}%: {v:.6f}")

        return {
            'all_corr': all_corr,
            'all_corr_remote': all_corr_remote,
        }

    def compute_per_layer_stats(self):
        """按层计算统计量"""
        layers = sorted(set(d['layer'] for d in self.corr_data))
        print("\n" + "=" * 60)
        print("Per-Layer Statistics")
        print("=" * 60)
        print(f"{'Layer':>6} | {'Mean':>10} | {'Std':>10} | {'P90':>10} | {'P95':>10}")
        print("-" * 60)

        for layer in layers:
            layer_corr = torch.cat([d['corr'].flatten() for d in self.corr_data if d['layer'] == layer])
            p90 = torch.quantile(layer_corr, 0.9)
            p95 = torch.quantile(layer_corr, 0.95)
            print(f"{layer:>6} | {layer_corr.mean():>10.4f} | {layer_corr.std():>10.4f} | {p90:>10.4f} | {p95:>10.4f}")

    def sweep_thresholds(self, thresholds=None, use_normalized=True):
        """扫描不同阈值下的选择比例"""
        if not self.corr_remote_data:
            print("No remote correlation data!")
            return

        if thresholds is None:
            thresholds = [0.001, 0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]

        print("\n" + "=" * 60)
        print(f"Threshold Sweep ({'Normalized' if use_normalized else 'Absolute'})")
        print("=" * 60)
        print(f"{'Threshold':>10} | {'Selected %':>12} | {'Avg per Head':>12}")
        print("-" * 50)

        results = []
        for th in thresholds:
            total_selected = 0
            total_count = 0

            for d in self.corr_remote_data:
                corr = d['corr_remote']  # [H, num_remote]
                if use_normalized:
                    corr_max = corr.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
                    corr_norm = corr / corr_max
                else:
                    corr_norm = corr

                selected = (corr_norm >= th).sum().item()
                total_selected += selected
                total_count += corr.numel()

            ratio = total_selected / total_count if total_count > 0 else 0
            avg_per_head = total_selected / (len(self.corr_remote_data) * corr.shape[0]) if self.corr_remote_data else 0

            print(f"{th:>10.2f} | {ratio * 100:>11.2f}% | {avg_per_head:>12.1f}")
            results.append({'threshold': th, 'ratio': ratio})

        return results

    def visualize_distribution(self, save_path="corr_distribution.png", use_remote=True):
        """可视化相关性分布"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping visualization")
            return

        data = self.corr_remote_data if use_remote else self.corr_data
        key = 'corr_remote' if use_remote else 'corr'

        if not data:
            print("No data to visualize!")
            return

        all_corr = torch.cat([d[key].flatten() for d in data]).numpy()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 整体直方图
        ax = axes[0, 0]
        ax.hist(all_corr, bins=100, density=True, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Correlation')
        ax.set_ylabel('Density')
        ax.set_title(f'Overall Distribution (n={len(all_corr):,})')
        ax.axvline(np.mean(all_corr), color='r', linestyle='--', label=f'Mean={np.mean(all_corr):.4f}')
        ax.legend()

        # 2. 按层的箱线图
        ax = axes[0, 1]
        layers = sorted(set(d['layer'] for d in data))
        layer_data = [torch.cat([d[key].flatten() for d in data if d['layer'] == l]).numpy() for l in layers]
        ax.boxplot(layer_data, labels=[str(l) for l in layers])
        ax.set_xlabel('Layer')
        ax.set_ylabel('Correlation')
        ax.set_title('Distribution by Layer')

        # 3. CDF
        ax = axes[1, 0]
        sorted_corr = np.sort(all_corr)
        cdf = np.arange(1, len(sorted_corr) + 1) / len(sorted_corr)
        ax.plot(sorted_corr, cdf)
        ax.set_xlabel('Correlation')
        ax.set_ylabel('CDF')
        ax.set_title('Cumulative Distribution')
        ax.grid(True, alpha=0.3)
        # 标记常用分位数
        for q in [0.5, 0.9, 0.95]:
            val = np.quantile(all_corr, q)
            ax.axhline(q, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(val, color='gray', linestyle=':', alpha=0.5)
            ax.annotate(f'P{int(q * 100)}={val:.3f}', (val, q), fontsize=8)

        # 4. Normalized 分布（用于阈值选择）
        ax = axes[1, 1]
        normalized_corrs = []
        for d in data:
            corr = d[key]  # [H, N]
            corr_max = corr.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
            corr_norm = (corr / corr_max).flatten().numpy()
            normalized_corrs.append(corr_norm)
        all_norm = np.concatenate(normalized_corrs)
        ax.hist(all_norm, bins=100, density=True, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Normalized Correlation (per-head max=1)')
        ax.set_ylabel('Density')
        ax.set_title('Normalized Distribution (for threshold selection)')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved distribution plot to {save_path}")
        plt.close()

    def visualize_heatmap(self, layer_idx=0, save_path="corr_heatmap.png"):
        """可视化特定层的相关性热力图"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available")
            return

        layer_data = [d for d in self.corr_data if d['layer'] == layer_idx]
        if not layer_data:
            print(f"No data for layer {layer_idx}")
            return

        # 取第一个 batch
        corr = layer_data[0]['corr'].numpy()  # [H, N]

        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(corr, aspect='auto', cmap='viridis')
        ax.set_xlabel('Delay τ (blocks)')
        ax.set_ylabel('Head')
        ax.set_title(f'Layer {layer_idx}: Correlation by Head and Delay')
        plt.colorbar(im, ax=ax, label='Correlation')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved heatmap to {save_path}")
        plt.close()


# 全局实例
_corr_collector = CorrelationStatsCollector()


def get_corr_collector():
    """获取全局收集器"""
    return _corr_collector
