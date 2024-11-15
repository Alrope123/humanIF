import torch
import fbgemm_gpu  # Import the fbgemm library

def test_fbgemm():
    try:
        # Set up random tensors to simulate a common use case for fbgemm
        batch_size, input_dim, output_dim = 32, 128, 64
        input_tensor = torch.randn(batch_size, input_dim)
        weights = torch.randn(input_dim, output_dim)

        # Quantize the input tensor and weights
        quantized_input = torch.quantize_per_tensor(input_tensor, scale=0.1, zero_point=0, dtype=torch.qint8)
        quantized_weights = torch.quantize_per_tensor(weights, scale=0.1, zero_point=0, dtype=torch.qint8)

        # Perform matrix multiplication using quantized tensors
        # `fbgemm_gpu.quantized_linear` uses fbgemm backend for quantized operations
        result = fbgemm_gpu.quantized_linear(quantized_input, quantized_weights)

        # Verify the result
        print("FBGEMM quantized linear operation result shape:", result.shape)
        print("FBGEMM operation output:", result)

        # Simple check to confirm functionality (result must be tensor and not None)
        assert result is not None and isinstance(result, torch.Tensor)
        print("FBGEMM test passed.")

    except Exception as e:
        print("FBGEMM test failed:", e)


# Run the test
test_fbgemm()

model_name = "meta-llama/Meta-Llama-3-8B"
quantization_config = FbgemmFp8Config()
quantized_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output = quantized_model.generate(**input_ids, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))