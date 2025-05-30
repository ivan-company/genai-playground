# GenAI playground

In this project we'll try different tools AI models can give us

## Installation

```bash
# First, create a virtual environment
python3 -m venv .venv
# Now we activate the environment
source .venv/bin/activate
# Now we install the dependencies
pip install -r requirements.txt
```

## Configuration

First, if you want to try the "dalle" models, you'll need an API key. Once you get one, you have to create a ".env" file on the root of the project and add the following line:

```bash
OPENAPI_API_KEY=<your_key>
```

If you want to use Gemini, you'll need a a Google API Key and then set it up in the env file:

```bash
GOOGLE_API_KEY=<your_key>
```

Once you have that, you can customize the output on of the tool in different ways:

- Inside the "input" folder, you have all the images that will be available for the tool. Whenever you do `python -m main -m sd -a outpaint -i sample_1` you are asking DALL-E to gather the image `input/sample_1.png`
- If you want to change prompting used on image generation or outpainting, just go to the `constants.py` file and change it

## Usage

That will be read by the code inside the "dalle" model to call the API.

To know what can you do with the tool, simply run:

```bash
python -m main -h
```

If you don't specify a certain argument, it will assume you want all of them. For example:

```bash
python -m main -m sd
```

will mean: "run all the actions over all the images for StableDiffusion"
