import json
import matplotlib.pyplot as plt

def generate_metrics_plot(metrics_path, output_path):
    """Generate a bar plot of metrics."""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.savefig(output_path)
    plt.close()
    print(f"Metrics plot saved to {output_path}")

if __name__ == "__main__":
    generate_metrics_plot('../reports/metrics.json', '../reports/metrics_bar.png')
