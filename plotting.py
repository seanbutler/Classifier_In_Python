"""
Plotting functions for training visualization
"""

import torch
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


def calculate_accuracy(model, data_loader):
    """
    Calculate accuracy on a dataset
    
    Args:
        model: The neural network model
        data_loader: DataLoader for the dataset
        
    Returns:
        accuracy as a percentage
    """
    correct = 0
    total = 0
    
    # Don't compute gradients for validation
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model.train_forward(images)
            # Get the predicted class (highest score)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def plot_training_curves(loss_history, accuracy_history, validation_accuracy_history, config, model, test_loader):
    """
    Plot enhanced training metrics with additional information
    
    Args:
        loss_history: List of loss values
        accuracy_history: List of test accuracy values
        validation_accuracy_history: List of validation accuracy values
        config: Configuration dictionary
        model: Trained neural network
        test_loader: Test data loader for confusion matrix
    """
    output_config = config['output']
    log_interval = config['training']['log_interval']
    
    # Create figure with subplot_mosaic layout
    mosaic = [
        ['loss','loss', 'accuracy','accuracy', 'config'],
        ['loss', 'loss', 'accuracy', 'accuracy', 'config'],
        ['loss_stats', 'improvement','weight_dist', 'class_acc', 'config'],
        ['d0', 'd1', 'd2', 'd3', 'd4'],
        ['d5', 'd6', 'd7', 'd8', 'd9']
    ]
    
    fig, axs = plt.subplot_mosaic(mosaic, 
                                   figsize=(20, 20),
                                   constrained_layout=True)
    
    # 1. Training Loss Plot
    ax1 = axs['loss']
    ax1.set_ylim(0.0, 1.0) 
    ax1.plot(loss_history, label='Training Loss', color='blue', linewidth=2)
    ax1.set_xlabel(f'Iteration (every {log_interval} batches)', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss Over Time', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add min/max annotations
    min_loss = min(loss_history)
    max_loss = max(loss_history)
    ax1.axhline(y=min_loss, color='green', linestyle='--', alpha=0.5, label=f'Min: {min_loss:.3f}')
    ax1.text(len(loss_history), min_loss, f'Min: {min_loss:.3f}', 
             verticalalignment='bottom', 
             horizontalalignment='right',
             fontsize=9, 
             color='green')
    
    # 2. Accuracy Plot
    ax2 = axs['accuracy']
    ax2.set_ylim(80.0, 100.0) 
    epochs = range(1, len(accuracy_history) + 1)
    ax2.plot(epochs, accuracy_history, label='Test Accuracy', color='green', linewidth=2, marker='o', markersize=6)
    ax2.plot(epochs, validation_accuracy_history, label='Validation Accuracy', color='blue', linewidth=2, marker='s', markersize=6)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Test & Validation Accuracy Over Time', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add final accuracy annotation
    final_acc = accuracy_history[-1]
    ax2.axhline(y=final_acc, color='red', linestyle='--', alpha=0.5, label=f' Final: {final_acc:.2f}%')
    ax2.text(len(epochs), final_acc, f' Final: {final_acc:.2f}%', 
             fontsize=9, 
             verticalalignment='top', 
             horizontalalignment='right',
             color='red' )

    # 3. Configuration Info Panel
    ax3 = axs['config']
    ax3.axis('off')
    
    final_val_acc = validation_accuracy_history[-1]
    config_text = f"""Architecture: {config['model']['hidden_layers']}
Epochs: {config['training']['epochs']}
Batch Size: {config['training']['batch_size']}
Learning Rate: {config['training']['learning_rate']}
Momentum: {config['training']['momentum']}
Final Test Acc: {final_acc:.2f}%
Final Val Acc: {final_val_acc:.2f}%
Best Test Acc: {max(accuracy_history):.2f}%
Best Val Acc: {max(validation_accuracy_history):.2f}%
Final Loss: {loss_history[-1]:.4f}
Min Loss: {min(loss_history):.4f}
"""

    ax3.text(0.05, 0.95, config_text, transform=ax3.transAxes, 
             fontsize=12,
             verticalalignment='top', 
            # fontfamily='monospace',
            #  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
             )
    
    # 4. Loss Statistics
    ax4 = axs['loss_stats']
    ax4.set_ylim(0.0, 3.0) 
    loss_stats = {
        'Min': min(loss_history),
        'Max': max(loss_history),
        'Mean': np.mean(loss_history),
        'Std': np.std(loss_history),
        'Final': loss_history[-1]
    }
    bars = ax4.bar(loss_stats.keys(), loss_stats.values(), color=['green', 'red', 'blue', 'orange', 'purple'])
    ax4.set_ylabel('Loss Value', fontsize=10)
    ax4.set_title('Loss Statistics', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 5. Accuracy Improvement
    ax5 = axs['improvement']
    ax5.set_ylim(-1.5, 8.0) 

    if len(accuracy_history) > 1:
        improvements = [accuracy_history[i] - accuracy_history[i-1] 
                       for i in range(1, len(accuracy_history))]
        ax5.bar(range(2, len(accuracy_history) + 1), improvements, 
               color=['green' if x > 0 else 'red' for x in improvements])
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax5.set_xlabel('Epoch', fontsize=10)
        ax5.set_ylabel('Accuracy Change (%)', fontsize=10)
        ax5.set_title('Epoch-to-Epoch Improvement', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Simple Confusion Matrix Preview (for 10 classes)
    ax6 = axs['class_acc']
    # Calculate simple per-class accuracy
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model.train_forward(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1
            break  # Just first batch for speed
    
    class_accuracy = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                     for i in range(10)]
    
    colors = ['green' if acc > 90 else 'orange' if acc > 70 else 'red' for acc in class_accuracy]
    ax6.barh(range(10), class_accuracy, color=colors, alpha=0.7)
    ax6.set_yticks(range(10))
    ax6.set_yticklabels([f'Digit {i}' for i in range(10)], fontsize=8)
    ax6.set_xlabel('Accuracy (%)', fontsize=10)
    ax6.set_title('Per-Class Accuracy (Sample)', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 9. Weight Distribution Histogram
    ax9 = axs['weight_dist']

    # # Add a single colorbar for all subplots
    all_weights = []
    for layer in model.layers:
        all_weights.extend(layer.weight.data.cpu().numpy().flatten())
    ax9.hist(all_weights, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax9.set_xlabel('Weight Value', fontsize=10)
    ax9.set_ylabel('Frequency', fontsize=10)
    ax9.set_title('Weight Distribution\n(All Layers)', fontsize=11, fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='y')
    ax9.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add statistics text
    mean_weight = np.mean(all_weights)
    std_weight = np.std(all_weights)
    ax9.text(0.95, 0.95, f'μ={mean_weight:.3f}\nσ={std_weight:.3f}', 
             transform=ax9.transAxes, fontsize=9, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 10. Output Layer Weight Patterns - Show effective weights for each digit
    # Compute effective weights from input to output by multiplying through all layers
    effective_weights = model.layers[-1].weight.data.cpu().numpy()
    
    # Multiply backwards through all hidden layers
    for layer in reversed(model.layers[:-1]):
        layer_weights = layer.weight.data.cpu().numpy()
        effective_weights = effective_weights @ layer_weights
    
    # Now effective_weights is [10, 784] - from input pixels to output classes
    # Visualize each digit's weight pattern
    for digit in range(10):
        ax_digit = axs[f'd{digit}']
        
        # Reshape the 784 input weights to 28x28 image
        digit_weights = effective_weights[digit, :].reshape(28, 28)
        
        im = ax_digit.imshow(digit_weights, cmap='RdBu', interpolation='nearest',
                            vmin=-np.abs(effective_weights).max(), 
                            vmax=np.abs(effective_weights).max())
        ax_digit.set_title(f'Digit {digit}', fontsize=10, fontweight='bold')
        ax_digit.set_xticks([])
        ax_digit.set_yticks([])
    
    # Add overall title
    fig.suptitle('MNIST Training Analysis Dashboard', fontsize=16, fontweight='bold')
    
    if output_config['save_plot']:
        # Add timestamp to filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = output_config['plot_filename']
        
        # Use study output directory if available, otherwise use default
        if 'study_output_dir' in config:
            plot_dir = f"{config['study_output_dir']}/plots"
        else:
            plot_dir = "outputs/plots"
        
        # Split filename and extension
        if '.' in base_filename:
            name, ext = base_filename.rsplit('.', 1)
            timestamped_filename = f"{plot_dir}/{name}_{timestamp}.{ext}"
        else:
            timestamped_filename = f"{plot_dir}/{base_filename}_{timestamp}"
        
        plt.savefig(timestamped_filename, dpi=300, bbox_inches='tight')
        print(f"\nEnhanced training metrics saved as '{timestamped_filename}'")
    
    if output_config['show_plot']:
        plt.show()
