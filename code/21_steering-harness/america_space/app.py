"""AmericaAI — satirical live demo of contrastive activation steering.

Consumes steering_bundle.pt (schema_version 1) produced by america_export.py
in the steering harness. Four independent directions, four sliders, five
presets. No fine-tuning, no system prompt — the sliders scale vectors added
directly to the residual stream of google/gemma-2-2b-it.

Runs on a Hugging Face Space with ZeroGPU. Secret required: HF_TOKEN
(gemma-2 is gated).
"""

import os
import threading

import gradio as gr
import torch

try:
    import spaces  # provided on ZeroGPU hardware

    ZERO_GPU = True
except ImportError:  # local / CPU fallback

    class spaces:
        @staticmethod
        def GPU(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda fn: fn

    ZERO_GPU = False

from transformers import TextIteratorStreamer

from runtime import SteeredGemma, load_bundle

BUNDLE_PATH = os.environ.get("BUNDLE_PATH", "steering_bundle.pt")
bundle = load_bundle(BUNDLE_PATH)

DEVICE = "cuda" if (ZERO_GPU or torch.cuda.is_available()) else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

engine = SteeredGemma(
    bundle,
    device=DEVICE,
    dtype=DTYPE,
    hf_token=os.environ.get("HF_TOKEN"),
)

ORDER = ["americana", "patriotic_pride", "trump_approval", "star_spangled_bombast"]

# One knob: FREEDOM LEVEL 0-500 maps to a uniform multiplier (100 -> 1.0x,
# the calibrated america_ai preset) across all four steering vectors.
# Legend marks are QA'd against the live model every 50 (see harness
# summary.md): ~200 the first flags appear, ~390 is overtly American but
# still coherent (the default), 500 dissolves into star-spangled word salad.
# Default is well above the harness-calibrated 1.0x, which reads as
# barely-steered in chat.
DEFAULT_FREEDOM = 390


@spaces.GPU(duration=90)
def chat(message, history, freedom):
    multiplier = float(freedom) / 100.0
    engine.set_strengths({name: multiplier for name in ORDER})

    msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ]
    msgs.append({"role": "user", "content": message})
    input_ids = engine.build_input_ids(msgs)

    streamer = TextIteratorStreamer(
        engine.tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    threading.Thread(
        target=engine.model.generate,
        kwargs=dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            # keep generation eager: steering hooks must fire, and torch 2.8
            # dynamo breaks on gemma-2's rope under compiled hybrid-cache decode
            disable_compile=True,
        ),
    ).start()

    partial = ""
    for token_text in streamer:
        partial += token_text
        yield partial


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Rye&family=Inter:wght@400;600&display=swap');
body, .gradio-container { background: #0A3161 !important; font-family: 'Inter', sans-serif; }
#title h1 { font-family: 'Rye', serif; text-align: center; color: #FFF; font-size: 3rem;
  letter-spacing: 0.04em; margin-bottom: 0; text-shadow: 3px 3px 0 #B31942; }
#tagline { text-align: center; color: #E8C547; letter-spacing: 0.25em;
  text-transform: uppercase; font-size: 0.8rem; margin-top: 4px; }
#stars { text-align: center; color: #FFF; letter-spacing: 0.6em; margin: 6px 0 14px; }
#howto { text-align: center; color: #FFF; font-size: 0.95rem; margin-bottom: 10px; }
#howto b { color: #E8C547; }
#freedom label span { color: #E8C547 !important; font-weight: 600; font-size: 1.15rem;
  letter-spacing: 0.12em; text-transform: uppercase; }
#legend { color: #9FB3D1; font-size: 0.85rem; text-align: center; line-height: 1.7; }
#legend strong { color: #E8C547; }
#disclaimer { color: #9FB3D1; font-size: 0.78rem; text-align: center; line-height: 1.5; }
#disclaimer a { color: #E8C547; }
"""

with gr.Blocks(css=CSS, title="AmericaAI") as demo:
    gr.HTML(
        '<div id="title"><h1>AMERICA AI</h1></div>'
        '<div id="tagline">American AI, for Americans</div>'
        '<div id="stars">&#9733; &#9733; &#9733; &#9733; &#9733;</div>'
        '<div id="howto">Already tuned to <b>peak America</b> &mdash; just ask it something. '
        "Crank the <b>FREEDOM LEVEL</b> to taste.</div>"
    )

    # rendered up top so it's the first thing users see; ChatInterface reuses it
    freedom = gr.Slider(
        0,
        500,
        value=DEFAULT_FREEDOM,
        step=5,
        label="🦅 FREEDOM LEVEL",
        elem_id="freedom",
    )
    gr.Markdown(
        "😐 **Normal model** &nbsp;·&nbsp; 🇺🇸 **Hints of America** &nbsp;·&nbsp; "
        "🦅 **Max Freedom** &nbsp;·&nbsp; 🥴 **Star Drunk**",
        elem_id="legend",
    )

    gr.ChatInterface(
        fn=chat,
        type="messages",
        additional_inputs=[freedom],
        concurrency_limit=1,  # hook state is shared; serialize requests
        cache_examples=False,
        examples=[
            ["What country will win the world cup?"],
            ["Give me a quick pep talk."],
            ["Who should I vote for?"],
        ],
    )

    with gr.Accordion("Wait, how does this actually work?", open=False):
        gr.Markdown(
            f"**TL;DR** — This is an unmodified `{bundle.target_model}`. No fine-tuning, "
            "no system prompt, no prompt tricks — pure steering vectors. I found four "
            "directions in the model's activation space — Americana, national pride, "
            "Trump approval, and star-spangled bombast — that add up to a composite "
            "*patriotism* concept. The FREEDOM LEVEL slider scales those vectors, which "
            "get added straight into the model's hidden state while it writes. At 0 "
            "you're talking to plain Gemma; crank it and the direction dominates until "
            "everything dissolves into star-spangled word salad.\n\n"
            "How the directions were found, and why this matters: "
            "**[read the full write-up](https://bitsofchris.com/p/american-ai)**.\n\n"
            "One person made this in half a day, and it's crude. Imagine what teams "
            "of experts with billions in funding can do to steer the models you use "
            "every day."
        )

    gr.HTML(
        '<div id="disclaimer">America AI is an intentionally politically steered '
        "parody. Its responses are generated entertainment, not neutral or factual "
        "guidance. Built as an educational demo of how easily language models can "
        "be steered.</div>"
    )

demo.launch()
