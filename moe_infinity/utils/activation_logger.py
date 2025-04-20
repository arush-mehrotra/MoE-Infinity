"""
Utilities for logging activations from Mixture of Experts models.
"""

import os
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Callable, Optional, Union, Any


class MoEActivationLogger:
    """Logger for capturing MoE activations and routing decisions."""
    
    def __init__(self, model=None):
        """
        Initialize the activation logger.
        
        Args:
            model: Optional model to register hooks on initialization
        """
        self.activations = defaultdict(dict)
        self.routing_decisions = {}
        self.hook_handles = {}
        
        if model is not None:
            self.register_hooks(model)
    
    def hook_fn(self, layer_name: str) -> Callable:
        """Create a hook function for a specific layer."""
        def _hook(expert_idx, inputs, outputs, token_indices, weights, routing_info):
            # Store basic info for this expert
            if expert_idx not in self.activations[layer_name]:
                self.activations[layer_name][expert_idx] = {
                    "inputs": [],
                    "outputs": [],
                    "token_indices": [],
                    "weights": []
                }
            
            # Capture expert activations
            self.activations[layer_name][expert_idx]["inputs"].append(inputs.detach().cpu())
            self.activations[layer_name][expert_idx]["outputs"].append(outputs.detach().cpu())
            
            # Capture token routing information
            if token_indices is not None:
                self.activations[layer_name][expert_idx]["token_indices"].append(token_indices.detach().cpu())
            if weights is not None:
                self.activations[layer_name][expert_idx]["weights"].append(weights.detach().cpu())
                
            # Store routing decisions once per layer
            if layer_name not in self.routing_decisions:
                self.routing_decisions[layer_name] = {
                    k: v.detach().cpu() for k, v in routing_info.items()
                }
                
        return _hook
    
    def register_hooks(self, model):
        """
        Register activation hooks on all MoE blocks in the model.
        
        Args:
            model: The model containing MoE blocks
        
        Returns:
            Dict of hook handles
        """
        from ..models.deepseek import DeepseekMoEBlock
        
        for name, module in model.named_modules():
            if isinstance(module, DeepseekMoEBlock):
                handle = module.register_activation_hook(
                    name=f"logger_{name}", 
                    hook_fn=self.hook_fn(name)
                )
                self.hook_handles[name] = handle
                
        return self.hook_handles
    
    def remove_hooks(self, model):
        """Remove all registered hooks."""
        from ..models.deepseek import DeepseekMoEBlock
        
        for name, module in model.named_modules():
            if isinstance(module, DeepseekMoEBlock) and name in self.hook_handles:
                module.remove_activation_hook(self.hook_handles[name])
        
        self.hook_handles = {}
    
    def clear(self):
        """Clear all stored activations and routing decisions."""
        self.activations = defaultdict(dict)
        self.routing_decisions = {}
    
    def save(self, save_dir: str, prefix: str = "moe_activations"):
        """
        Save activations and routing decisions to disk.
        
        Args:
            save_dir: Directory to save files
            prefix: Prefix for filenames
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save activations
        activation_path = os.path.join(save_dir, f"{prefix}_activations.pt")
        torch.save(self.activations, activation_path)
        
        # Save routing decisions
        routing_path = os.path.join(save_dir, f"{prefix}_routing.pt")
        torch.save(self.routing_decisions, routing_path)
        
        return activation_path, routing_path
    
    @staticmethod
    def load(activation_path: str, routing_path: str = None):
        """
        Load previously saved activations and routing decisions.
        
        Args:
            activation_path: Path to the saved activations
            routing_path: Path to the saved routing decisions
            
        Returns:
            MoEActivationLogger instance with loaded data
        """
        logger = MoEActivationLogger()
        logger.activations = torch.load(activation_path)
        
        if routing_path is not None:
            logger.routing_decisions = torch.load(routing_path)
        
        return logger
        
    def get_expert_usage(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate expert usage statistics.
        
        Returns:
            Dict of layer -> expert -> usage percentage
        """
        usage_stats = {}
        
        for layer_name, experts in self.activations.items():
            usage_stats[layer_name] = {}
            total_tokens = 0
            
            # Count tokens per expert
            expert_counts = {}
            for expert_idx, data in experts.items():
                if expert_idx == "shared":
                    continue
                
                token_count = sum(indices.sum().item() for indices in data["token_indices"])
                expert_counts[expert_idx] = token_count
                total_tokens += token_count
            
            # Calculate percentages
            if total_tokens > 0:
                for expert_idx, count in expert_counts.items():
                    usage_stats[layer_name][str(expert_idx)] = count / total_tokens
            
        return usage_stats
    
    def get_routing_distribution(self) -> Dict[str, np.ndarray]:
        """
        Get the distribution of token routing across experts for each layer.
        
        Returns:
            Dict of layer -> distribution array
        """
        distributions = {}
        
        for layer_name, routing in self.routing_decisions.items():
            if "topk_idx" in routing:
                # Count occurrences of each expert
                topk_idx = routing["topk_idx"].numpy()
                unique, counts = np.unique(topk_idx, return_counts=True)
                
                # Create distribution array (normalized)
                num_experts = max(unique) + 1
                distribution = np.zeros(num_experts)
                distribution[unique] = counts
                distribution = distribution / distribution.sum()
                
                distributions[layer_name] = distribution
        
        return distributions


def example_usage():
    """Example of how to use the MoEActivationLogger."""
    # Assume model is your MoE model
    model = None  # Replace with your model
    
    # Create logger and register hooks
    logger = MoEActivationLogger(model)
    
    # Run inference or training
    inputs = None  # Replace with your inputs
    outputs = model(inputs)
    
    # Save activations
    logger.save("./moe_logs")
    
    # Print expert usage statistics
    expert_usage = logger.get_expert_usage()
    for layer, stats in expert_usage.items():
        print(f"Layer: {layer}")
        for expert, usage in stats.items():
            print(f"  Expert {expert}: {usage:.2%}")
    
    # Clean up
    logger.remove_hooks(model) 