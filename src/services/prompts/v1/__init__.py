import os
import chevron

PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_prompt(template_name: str, context: dict = None) -> str:
    template_path = os.path.join(PROMPTS_DIR, f"{template_name}.mustache")

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    if context:
        return chevron.render(template, context)

    return chevron.render(template, {})