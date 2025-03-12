import argparse
from rich import print
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()


console = Console()


if __name__ == '__main__':
    # First we read the user input
    parser = argparse.ArgumentParser(
        description='GenAI image generation/edition')
    parser.add_argument('-m', '--model', type=str, choices=["dalle", "sd", "all"], default="all",
                        help='The model to use for the generation')
    parser.add_argument('-a', '--action', type=str, choices=["generate", "outpaint", "all"], default="all",
                        help='The action to perform')
    args = parser.parse_args()

    # Then we execute the corresponding code
    if args.model == "dalle" or args.model == "all":
        if args.action == "generate" or args.action == "all":
            from dalle.generate import run
            print("Generating image with DALL-E")
            with console.status("[bold green]Generating..."):
                run()
        if args.action == "outpaint" or args.action == "all":
            from dalle.outpaint import run
            print("Outpainting image with DALL-E")
            with console.status("[bold green]Outpainting..."):
                run()
    if args.model == "sd" or args.model == "all":
        if args.action == "generate" or args.action == "all":
            from sd.generate import run
            print("Generating image with Stable Diffusion")
            with console.status("[bold green]Generating..."):
                run()
        if args.action == "outpaint" or args.action == "all":
            from sd.outpaint import run
            print("Outpainting image with Stable Diffusion")
            with console.status("[bold green]Outpainting..."):
                run()

    print("Done!")
