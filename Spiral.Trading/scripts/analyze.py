#!/usr/bin/env python3
"""Analyze model predictions - confusion matrix and visualizations."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import TradingDataset, RowGroupSampler

TREND_NAMES = [
    'StrongUptrend',
    'MidUptrend',
    'WeakUptrend',
    'Consolidation',
    'WeakDowntrend',
    'MidDowntrend',
    'StrongDowntrend'
]


def load_mixer_model(path='data/mixer_v3_model.pt'):
    from train_mixer_v3 import TradingMixerV3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TradingMixerV3().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model, device


def predict_batch(model, batch, device):
    """Model-agnostic batch prediction. Returns trend logits."""
    x_1s = batch['features_1s'].to(device)
    x_1m = batch['features_1m'].to(device)
    x_5m = batch['features_5m'].to(device)
    _, trend_logits = model(x_1s, x_1m, x_5m)
    return trend_logits


def get_predictions(model, loader, device, max_batches=None):
    """Get all predictions and labels from a loader."""
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            
            trend_labels = batch['trend']
            trend_logits = predict_batch(model, batch, device)
            probs = torch.softmax(trend_logits, dim=1)
            preds = trend_logits.argmax(1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(trend_labels.numpy())
            all_probs.append(probs.cpu().numpy())
    
    return (np.concatenate(all_preds), 
            np.concatenate(all_labels),
            np.concatenate(all_probs))


def plot_confusion_matrix(preds, labels, save_path='data/confusion_matrix.png'):
    """Plot and save confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(labels, preds, labels=range(7))
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    im1 = axes[0].imshow(cm, cmap='Blues')
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_xticks(range(7))
    axes[0].set_yticks(range(7))
    axes[0].set_xticklabels(TREND_NAMES, rotation=45, ha='right')
    axes[0].set_yticklabels(TREND_NAMES)
    plt.colorbar(im1, ax=axes[0])
    
    # Normalized (percentages)
    im2 = axes[1].imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_xticks(range(7))
    axes[1].set_yticks(range(7))
    axes[1].set_xticklabels(TREND_NAMES, rotation=45, ha='right')
    axes[1].set_yticklabels(TREND_NAMES)
    plt.colorbar(im2, ax=axes[1])
    
    # Add percentage text
    for i in range(7):
        for j in range(7):
            val = cm_normalized[i, j]
            color = 'white' if val > 0.5 else 'black'
            axes[1].text(j, i, f'{val:.1%}', ha='center', va='center', color=color, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved confusion matrix to {save_path}")
    plt.close()
    
    return cm, cm_normalized


def plot_price_with_predictions(dataset, model, device, day_idx=0, stride=60, save_path='data/price_chart.png'):
    """Plot price chart with model predictions overlaid.
    
    Args:
        stride: Prediction stride for visualization (default 60 = 1 per minute for cleaner charts)
    """
    dataset._load_row_group(day_idx)
    data = dataset._cached_data
    
    prices = data['close']
    true_trends = data['trend']
    times = np.arange(len(prices)) / 60  # Convert to minutes
    
    # Get model predictions at each stride point
    pred_times = []
    pred_trends = []
    pred_probs = []
    
    model.eval()
    with torch.no_grad():
        for pos in range(0, len(prices), stride):
            sample = dataset.get_features_for(day_idx, pos)
            x_1s = sample['features_1s'].unsqueeze(0).to(device)
            x_1m = sample['features_1m'].unsqueeze(0).to(device)
            x_5m = sample['features_5m'].unsqueeze(0).to(device)
            
            _, trend_logits = model(x_1s, x_1m, x_5m)
            probs = torch.softmax(trend_logits, dim=1).cpu().numpy()[0]
            pred = trend_logits.argmax(1).item()
            
            pred_times.append(pos / 60)
            pred_trends.append(pred)
            pred_probs.append(probs)
    
    pred_times = np.array(pred_times)
    pred_trends = np.array(pred_trends)
    pred_probs = np.array(pred_probs)
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Plot 1: Price with true trend colors
    ax1 = axes[0]
    trend_colors = ['#d62728', '#ff7f0e', '#ffbb78', '#7f7f7f', '#98df8a', '#2ca02c', '#1f77b4']
    
    for i in range(len(prices) - 1):
        ax1.plot([times[i], times[i+1]], [prices[i], prices[i+1]], 
                 color=trend_colors[true_trends[i]], linewidth=0.5)
    ax1.set_ylabel('Price')
    ax1.set_title(f'Day {day_idx}: Price with True Trend Labels')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=trend_colors[i], label=TREND_NAMES[i]) for i in range(7)]
    ax1.legend(handles=legend_elements, loc='upper left', ncol=4, fontsize=8)
    
    # Plot 2: Price with predicted trend colors
    ax2 = axes[1]
    for i in range(len(pred_times) - 1):
        start_idx = int(pred_times[i] * 60)
        end_idx = int(pred_times[i+1] * 60)
        ax2.plot(times[start_idx:end_idx+1], prices[start_idx:end_idx+1],
                 color=trend_colors[pred_trends[i]], linewidth=0.5)
    ax2.set_ylabel('Price')
    ax2.set_title('Price with Predicted Trend Labels')
    
    # Plot 3: Prediction probabilities over time
    ax3 = axes[2]
    for i, name in enumerate(TREND_NAMES):
        ax3.plot(pred_times, pred_probs[:, i], label=name, color=trend_colors[i], alpha=0.7)
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Probability')
    ax3.set_title('Trend Prediction Probabilities')
    ax3.legend(loc='upper left', ncol=4, fontsize=8)
    ax3.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved price chart to {save_path}")
    plt.close()


def main():
    print("Loading model...")
    model, device = load_mixer_model()
    
    print("Loading test dataset...")
    test_dataset = TradingDataset('data/test.parquet', window_size=60, stride=5)
    test_loader = DataLoader(
        test_dataset, batch_size=1024, 
        sampler=RowGroupSampler(test_dataset, shuffle=False),
        num_workers=4
    )
    
    print("Getting predictions (first 500 batches)...")
    preds, labels, probs = get_predictions(model, test_loader, device, max_batches=500)
    print(f"Got {len(preds):,} predictions")
    
    print("\nGenerating confusion matrix...")
    cm, cm_norm = plot_confusion_matrix(preds, labels)
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for i, name in enumerate(TREND_NAMES):
        acc = cm_norm[i, i]
        print(f"  {name:20s}: {acc:.1%}")
    
    print("\nGenerating price charts for 3 sample days...")
    for day_idx in [0, 100, 500]:
        save_path = f'data/price_chart_day{day_idx}.png'
        plot_price_with_predictions(test_dataset, model, device, day_idx, stride=60, save_path=save_path)


if __name__ == "__main__":
    main()
