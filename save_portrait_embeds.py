import torch
from diffusers import LTXImageToVideoPipeline

from preview_prompts import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PREVIEW_PROMPT

MODEL_ID = "Lightricks/LTX-Video"
MODEL_REVISION = "8984fa25007f376c1a299016d0957a37a2f797bb"
OUT_PATH = "portrait_prompt_embeds.pt"
DTYPE = torch.bfloat16
MAX_SEQUENCE_LENGTH = 256

prompt = DEFAULT_PREVIEW_PROMPT
negative_prompt = DEFAULT_NEGATIVE_PROMPT

pipe = LTXImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    torch_dtype=DTYPE,
).to("cuda")
if pipe.tokenizer is not None:
    pipe.tokenizer.model_max_length = MAX_SEQUENCE_LENGTH

with torch.inference_mode():
    encoded = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        device=torch.device("cuda"),
        dtype=DTYPE,
    )

prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = encoded

torch.save(
    {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "prompt_embeds": prompt_embeds.detach().cpu(),
        "prompt_attention_mask": prompt_attention_mask.detach().cpu(),
        "negative_prompt_embeds": negative_prompt_embeds.detach().cpu(),
        "negative_prompt_attention_mask": negative_prompt_attention_mask.detach().cpu(),
    },
    OUT_PATH,
)
print(f"saved {OUT_PATH}: prompt_embeds={tuple(prompt_embeds.shape)}")
