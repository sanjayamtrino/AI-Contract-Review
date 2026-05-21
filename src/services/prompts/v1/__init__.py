import os
from typing import Optional

import pystache

PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))

_renderer = pystache.Renderer(escape=lambda u: u)


def load_prompt(template_name: str, context: Optional[dict] = None) -> str:
    template_path = os.path.join(PROMPTS_DIR, f"{template_name}.mustache")

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    if context:
        return _renderer.render(template, context)

    return template
