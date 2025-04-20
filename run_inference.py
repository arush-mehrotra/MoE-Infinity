import torch
import os
from transformers import AutoTokenizer
from moe_infinity import MoE

user_home = os.path.expanduser('~')

checkpoint = "deepseek-ai/DeepSeek-V2-Lite-Chat"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

config = {
    "offload_path": os.path.join(user_home, "moe-offload-dir"),  # local SSD path
    "device_memory_ratio": 0.75,
}

model = MoE(checkpoint, config)

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda:0")

output_ids = model.generate(input_ids)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated output:", output_text)

# Print logs for each MoE layer
for layer_id, moe_block in enumerate(model.moe_blocks):  # Assuming model.moe_blocks exists
    print(f"\nStats for MoE Layer {layer_id}:")
    layer_stats = moe_block.get_layer_statistics()
    if layer_stats:
        # Print available statistics
        for stat_name, stat_value in layer_stats.items():
            print(f"{stat_name}: {stat_value}")
        
    # If you want to see the raw activation logs
    activation_logs = moe_block.get_activation_logs()
    print("\nDetailed activation logs for this layer:", activation_logs)
    
    # Clear logs after printing if you want
    moe_block.clear_logs()
