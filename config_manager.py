"""
Configuration management for MNIST training
"""

import json


def load_config(config_path='config.json'):
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config, config_path='config.json'):
    """
    Save configuration to JSON file
    
    Args:
        config: Dictionary containing configuration parameters
        config_path: Path to save the configuration JSON file
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def create_config(model_params=None, training_params=None, data_params=None, output_params=None):
    """
    Create a configuration dictionary with specified parameters
    
    Args:
        model_params: Dictionary with model configuration (input_size, hidden_layers, output_size)
        training_params: Dictionary with training configuration (epochs, batch_size, learning_rate, momentum, log_interval)
        data_params: Dictionary with data configuration (data_root, normalize_mean, normalize_std, etc.)
        output_params: Dictionary with output configuration (save_plot, plot_filename, show_plot)
        
    Returns:
        Complete configuration dictionary
    """
    config = {
        'model': model_params or {
            'input_size': 784,
            'hidden_layers': [64, 32],
            'output_size': 10
        },
        'training': training_params or {
            'epochs': 10,
            'batch_size': 64,
            'learning_rate': 0.001,
            'momentum': 0.9,
            'log_interval': 100
        },
        'data': data_params or {
            'data_root': './data',
            'normalize_mean': 0.5,
            'normalize_std': 0.5,
            'test_batch_size': 100,
            'shuffle_train': True,
            'shuffle_test': False
        },
        'output': output_params or {
            'save_plot': True,
            'plot_filename': 'training_metrics.png',
            'show_plot': True
        }
    }
    return config


def update_config(config, **kwargs):
    """
    Update specific configuration parameters
    
    Args:
        config: Existing configuration dictionary
        **kwargs: Key-value pairs to update (e.g., learning_rate=0.01, hidden_layers=[128, 64])
        
    Returns:
        Updated configuration dictionary
    """
    # Map simple parameter names to their nested locations
    param_map = {
        'hidden_layers': ('model', 'hidden_layers'),
        'input_size': ('model', 'input_size'),
        'output_size': ('model', 'output_size'),
        'epochs': ('training', 'epochs'),
        'batch_size': ('training', 'batch_size'),
        'learning_rate': ('training', 'learning_rate'),
        'momentum': ('training', 'momentum'),
        'log_interval': ('training', 'log_interval'),
    }
    
    for param, value in kwargs.items():
        if param in param_map:
            section, key = param_map[param]
            config[section][key] = value
        else:
            # Handle direct nested access like model__hidden_layers
            if '__' in param:
                section, key = param.split('__', 1)
                if section in config:
                    config[section][key] = value
    
    return config
