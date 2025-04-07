import torch
import gradio as gr
import requests
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


print(torch.cuda.is_available())
def generate_response(model, processor, image, question):

    # Create input messages
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        },
    ]

    print("Preparing inputs!")
    # Prepare inputs
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to("cuda")

    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=100)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    
    final_answer = generated_texts[0].split('assistant', 1)[-1].strip()
    
    return final_answer


def main(capture,prompt):
    print("Loading Model")
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
      model_id, 
      torch_dtype=torch.float16, 
      low_cpu_mem_usage=True,).to("cuda")

    processor = AutoProcessor.from_pretrained(model_id)

    print("Generating Response")
    response = generate_response(model, processor, capture, prompt)
    return response


with gr.Blocks() as demo:
    with gr.Row():
            gr.Markdown(
                "<h1 style='text-align: center; color: black; size : 20px;'>Image Analysis using LlaVa-OneVision</h1>"
            )

    with gr.Row():
        with gr.Column(scale=2):
            capture = gr.Image(label="Capture a Scene", sources=["webcam"], type="numpy")

        with gr.Column(scale=1):
            with gr.Row():
                prompt = gr.Text(label="Prompt")
            with gr.Row():
                gosignal = gr.Button(value="Submit Prompt")
            with gr.Row():
                answer = gr.Text(label="Answer")

    gosignal.click(main,[capture,prompt],answer)


if __name__ == "__main__":
    try:
        demo.launch()
    except KeyboardInterrupt:
        print("Stopping the application...")
