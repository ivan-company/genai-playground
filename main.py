import argparse
import importlib.util
import sys
from pathlib import Path
from rich import print
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()
console = Console()


def call_dynamic_function(model, action, *args, **kwargs):
    module_path = f"models/{model}/{action}.py"
    try:
        # Load the module specification
        spec = importlib.util.spec_from_file_location(f"{model}_{action}", module_path)
        if spec is None:
            raise ImportError(f"Could not find module at {module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{model}_{action}"] = module

        spec.loader.exec_module(module)

        if not hasattr(module, "run"):
            raise AttributeError(f"Module {module_path} does not have a 'run' function")

        return module.run(*args, **kwargs)

    except (ImportError, AttributeError) as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    available_models = [f.name for f in Path("models").iterdir() if f.is_dir()]
    available_actions = [
        f.name.replace(".py", "")
        for f in Path("models/sd").iterdir()
        if f.suffix == ".py" and f.name != "__init__.py"
    ]
    available_images = [
        f.name.replace(".png", "")
        for f in Path("input").iterdir()
        if f.suffix == ".png"
    ]
    parser = argparse.ArgumentParser(description="GenAI image generation/edition")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=available_models + ["all"],
        default="all",
        help="The model to use for the generation",
    )
    parser.add_argument(
        "-a",
        "--action",
        type=str,
        choices=available_actions + ["all"],
        default="all",
        help="The action to perform",
    )
    parser.add_argument(
        "-i", "--images", type=str, default="all", help="The image to process"
    )
    args = parser.parse_args()

    models, actions, images = [], [], []

    if args.model == "all":
        models = available_models
    else:
        models = [args.model]

    if args.action == "all":
        actions = available_actions
    else:
        actions = [args.action]

    if args.images == "all":
        images = available_images
    else:
        images = [args.images]

    for model in models:
        for action in actions:
            with console.status(f"Processing {model} {action}..."):
                call_dynamic_function(
                    model, action, image_names=images, output_prefix=f"{model}_{action}"
                )
            console.print("Done!")
