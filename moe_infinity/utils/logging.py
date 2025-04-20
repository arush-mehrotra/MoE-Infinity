import torch
from typing import Dict, List
import numpy as np

class MoELogger:
    def __init__(self):
        self.layer_logs = {}
        
    def log_layer(self, layer_id: int, 
                  router_logits: torch.Tensor,
                  routing_weights: torch.Tensor,
                  selected_experts: torch.Tensor,
                  hidden_states: torch.Tensor,
                  expert_outputs: Dict[int, torch.Tensor] = None):
        """
        Log metrics for a specific layer
        
        Args:
            layer_id: ID of the current layer
            router_logits: Raw logits from router (batch_size * seq_len, n_experts) or None
            routing_weights: Normalized weights for selected experts
            selected_experts: Expert indices selected for each token
            hidden_states: Hidden states for each token
            expert_outputs: Optional dict mapping expert ID to its output
        """
        with torch.no_grad():
            if layer_id not in self.layer_logs:
                self.layer_logs[layer_id] = {
                    'router_logits': [],
                    'routing_weights': [],
                    'selected_experts': [],
                    'hidden_states': [],
                    'expert_outputs': []
                }
            
            # Store metrics
            if router_logits is not None:
                self.layer_logs[layer_id]['router_logits'].append(router_logits.cpu().numpy())
            self.layer_logs[layer_id]['routing_weights'].append(routing_weights.cpu().numpy())
            self.layer_logs[layer_id]['selected_experts'].append(selected_experts.cpu().numpy())
            self.layer_logs[layer_id]['hidden_states'].append(hidden_states.cpu().numpy())
            
            if expert_outputs:
                self.layer_logs[layer_id]['expert_outputs'].append(
                    {k: v.cpu().numpy() for k, v in expert_outputs.items()}
                )

    def get_layer_stats(self, layer_id: int):
        """Get statistics for a specific layer"""
        if layer_id not in self.layer_logs:
            return None
            
        logs = self.layer_logs[layer_id]
        
        # Compute statistics
        stats = {
            'routing_weights_mean': np.mean(logs['routing_weights']),
            'routing_weights_std': np.std(logs['routing_weights']),
            'expert_selection_counts': np.bincount(
                np.concatenate(logs['selected_experts']).flatten()
            ),
            'hidden_states_mean': np.mean(logs['hidden_states']),
            'hidden_states_std': np.std(logs['hidden_states'])
        }
        
        # Add router_logits stats if available
        if logs['router_logits']:
            stats['router_logits_mean'] = np.mean(logs['router_logits'])
            stats['router_logits_std'] = np.std(logs['router_logits'])
        
        return stats

    def clear(self):
        """Clear all logged data"""
        self.layer_logs = {} 