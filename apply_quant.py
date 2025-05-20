from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import oneshot
# Select quantization algorithm. In this case, we:
#   * apply SmoothQuant to make the activations easier to quantize
#   * quantize the weights to int8 with GPTQ (static per channel)
#   * quantize the activations to int8 (dynamic per token)
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
]
# Apply quantization using the built in open_platypus dataset.
oneshot(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    dataset="open_platypus",
    recipe=recipe,
    output_dir="Meta-Llama-3.1-8B-Instruct-INT8",
    max_seq_length=2048,
    num_calibration_samples=512,
)
