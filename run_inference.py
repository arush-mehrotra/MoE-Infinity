import torch
import os
from transformers import AutoTokenizer
from moe_infinity import MoE
from moe_infinity.utils.activation_logger import MoEActivationLogger  # Adjust path if necessary

user_home = os.path.expanduser('~')

checkpoint = "deepseek-ai/DeepSeek-V2-Lite-Chat"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

config = {
    "offload_path": os.path.join(user_home, "moe-offload-dir"),
    "device_memory_ratio": 0.75,
}

try:
    # Load model
    print("Loading model...")
    model = MoE(checkpoint, config)

    # ✅ Step 1: Register logger hooks
    print("Registering activation logger...")
    logger = MoEActivationLogger(model)

    # Prepare input
    input_text = "translate English to German: How old are you?"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda:0")
    print(f"Input text: {input_text}")

    # Run inference
    print("Running model inference...")
    output_ids = model.generate(input_ids)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"Model output: {output_text}")

    # ✅ Step 2: Save activations and routing decisions
    print("Saving activations and routing data...")
    log_dir = os.path.join(user_home, "moe-logs")
    try:
        saved_paths = logger.save(log_dir)
        if isinstance(saved_paths, tuple) and len(saved_paths) >= 2:
            activation_path, routing_path = saved_paths[:2]
            print(f"Activations saved to {activation_path}")
            print(f"Routing decisions saved to {routing_path}")
    except Exception as e:
        print(f"Warning: Failed to save activations: {e}")

    # ✅ Step 3: Print expert usage statistics
    print("\nCalculating expert usage statistics...")
    try:
        usage = logger.get_expert_usage()
        for layer, stats in usage.items():
            print(f"\n[Layer: {layer}]")
            for expert, percent in stats.items():
                print(f"  Expert {expert}: {percent:.2%}")
    except Exception as e:
        print(f"Warning: Failed to calculate expert usage: {e}")

    # ✅ Step 4: Optionally remove hooks
    print("\nRemoving activation hooks...")
    logger.remove_hooks(model)
    
except Exception as e:
    print(f"Error during execution: {e}")
    import traceback
    traceback.print_exc()
