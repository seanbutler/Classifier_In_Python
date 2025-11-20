"""
Experiment module for running MNIST training experiments
"""

import torch
from torchvision import transforms as tv_transforms
from torchvision import datasets as tv_datasets
from neural_network import NeuralNetwork
from plotting import calculate_accuracy, plot_training_curves


def run_experiment(config, experiment_name=None, verbose=True):
    """
    Run a single training experiment with the given configuration
    
    Args:
        config: Configuration dictionary with model, training, data, and output params
        experiment_name: Optional name for this experiment (used in output files)
        verbose: Whether to print detailed progress information
        
    Returns:
        Dictionary containing experiment results:
            - final_test_accuracy
            - final_validation_accuracy
            - best_test_accuracy
            - best_validation_accuracy
            - final_loss
            - min_loss
            - loss_history
            - accuracy_history
            - validation_accuracy_history
            - model (trained neural network)
    """
    
    if verbose:
        print(f"\n{'='*60}")
        if experiment_name:
            print(f"Running Experiment: {experiment_name}")
        print(f"{'='*60}")
        print(f"Model: {config['model']['hidden_layers']}")
        print(f"Training: {config['training']['epochs']} epochs, batch size {config['training']['batch_size']}")
        print(f"Learning Rate: {config['training']['learning_rate']}, Momentum: {config['training']['momentum']}")
        print()

    # Check GPU availability
    if verbose and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total VRAM: {total_memory:.1f} GB")
        print()
    elif verbose:
        print("Running on CPU")
        print()

    # Create transform from config
    data_config = config['data']
    transform = tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Normalize((data_config['normalize_mean'],), (data_config['normalize_std'],))
    ])

    # Load datasets from config
    train_data_full = tv_datasets.MNIST(root=data_config['data_root'], 
                                         train=True, 
                                         download=True, 
                                         transform=transform)

    test_data = tv_datasets.MNIST(root=data_config['data_root'],
                                   train=False,
                                   download=True,
                                   transform=transform)

    # Split training data into train and validation sets (80/20 split)
    train_size = int(0.8 * len(train_data_full))
    val_size = len(train_data_full) - train_size
    train_data, val_data = torch.utils.data.random_split(train_data_full, [train_size, val_size])
    
    if verbose:
        print(f"Training samples: {train_size}, Validation samples: {val_size}")

    # Create data loaders from config
    train_config = config['training']
    train_loader = torch.utils.data.DataLoader(train_data, 
                                               batch_size=train_config['batch_size'], 
                                               shuffle=data_config['shuffle_train'])

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=data_config['test_batch_size'],
                                             shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=data_config['test_batch_size'],
                                              shuffle=data_config['shuffle_test'])

    # Create network from config
    model_config = config['model']
    model = NeuralNetwork(
        input_size=model_config['input_size'],
        hidden_layers=model_config['hidden_layers'],
        output_size=model_config['output_size']
    )
    
    if verbose:
        print("Neural Network Architecture:")
        print(model)
        print()

    # Create optimizer and loss from config
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=train_config['learning_rate'], 
        momentum=train_config['momentum']
    )

    # Lists to track metrics
    loss_history = []
    accuracy_history = []
    validation_accuracy_history = []

    # Training loop
    for epoch in range(train_config['epochs']):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model.train_forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            log_interval = train_config['log_interval']
            if i % log_interval == log_interval - 1:
                avg_loss = running_loss / log_interval
                if verbose:
                    print(f"[{epoch + 1}, {i + 1}] loss: {avg_loss:.3f}")
                loss_history.append(avg_loss)
                running_loss = 0.0
        
        # Calculate accuracy at the end of each epoch
        test_accuracy = calculate_accuracy(model, test_loader)
        val_accuracy = calculate_accuracy(model, val_loader)
        accuracy_history.append(test_accuracy)
        validation_accuracy_history.append(val_accuracy)
        
        if verbose:
            print(f"Epoch {epoch + 1} - Test Accuracy: {test_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")
    
    # Generate plot if requested
    if config['output']['save_plot'] or config['output']['show_plot']:
        # Update plot filename if experiment name provided
        if experiment_name:
            original_filename = config['output']['plot_filename']
            if '.' in original_filename:
                name, ext = original_filename.rsplit('.', 1)
                config['output']['plot_filename'] = f"{name}_{experiment_name}.{ext}"
            else:
                config['output']['plot_filename'] = f"{original_filename}_{experiment_name}"
        
        plot_training_curves(loss_history, accuracy_history, validation_accuracy_history, 
                           config, model, test_loader)
    
    # Compile results
    results = {
        'experiment_name': experiment_name,
        'config': config,
        'final_test_accuracy': accuracy_history[-1],
        'final_validation_accuracy': validation_accuracy_history[-1],
        'best_test_accuracy': max(accuracy_history),
        'best_validation_accuracy': max(validation_accuracy_history),
        'final_loss': loss_history[-1] if loss_history else None,
        'min_loss': min(loss_history) if loss_history else None,
        'loss_history': loss_history,
        'accuracy_history': accuracy_history,
        'validation_accuracy_history': validation_accuracy_history,
        'model': model
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment Complete: {experiment_name if experiment_name else 'Unnamed'}")
        print(f"Final Test Accuracy: {results['final_test_accuracy']:.2f}%")
        print(f"Final Validation Accuracy: {results['final_validation_accuracy']:.2f}%")
        print(f"Best Test Accuracy: {results['best_test_accuracy']:.2f}%")
        print(f"{'='*60}\n")
    
    return results
