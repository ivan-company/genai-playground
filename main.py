import argparse
import importlib.util
import sys
import os
from rich import print
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()


console = Console()


def call_dynamic_function(model, action, *args, **kwargs):
    module_path = f"{model}/{action}.py"
    try:
        # Load the module specification
        spec = importlib.util.spec_from_file_location(
            f"{model}_{action}", module_path)
        if spec is None:
            raise ImportError(f"Could not find module at {module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{model}_{action}"] = module

        spec.loader.exec_module(module)

        if not hasattr(module, "run"):
            raise AttributeError(
                f"Module {module_path} does not have a 'run' function")

        return module.run(*args, **kwargs)

    except (ImportError, AttributeError) as e:
        print(f"Error: {e}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GenAI image generation/edition')
    parser.add_argument('-m', '--model', type=str, choices=["dalle", "sd", "all"], default="all",
                        help='The model to use for the generation')
    parser.add_argument('-a', '--action', type=str, choices=["generate", "outpaint", "background", "all"], default="all",
                        help='The action to perform')
    parser.add_argument('-i', '--images', type=str, default="all")
    args = parser.parse_args()

    if args.model == "all":
        models = ["dalle", "sd"]
    else:
        models = [args.model]

    if args.action == "all":
        actions = ["generate", "outpaint", "background"]
    else:
        actions = [args.action]

    if args.images == "all":
        args.images = []
        for filename in os.listdir("input"):
            if filename.endswith(".png"):
                args.images.append(filename.replace(".png", ""))
    else:
        args.images = [args.images]

    for model in models:
        for action in actions:
            console.print(f"Running {model} {action}...")
            call_dynamic_function(model, action, args.images)
            console.print(f"Finished {model} {action}...")
