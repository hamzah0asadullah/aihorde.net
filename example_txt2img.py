import aihorde_net as     horde
from   json        import dumps
from   sys         import exit
from   time        import time

# Note: For faster responses while chatting, set your API-key using horde.api_key = "apikeyhere"
# Assuming that, by wonder, the AI Horde's text-to-image section isn't under heavy load, that's required.

models: tuple[list[Exception], bool] | tuple[list[str] | bool] = horde.fetch_models(
    model_type="txt2img",         # choose from txt2txt and txt2img
    return_raw=False,             # return True uses more bandwidth and is generally only useful in very rare cases
    min_threads=None,             # Per documentation: "Filter only models that have at least this amount of threads serving." Only god knows that this actually refers too.
    max_threads=None,             # Opposite of min_threads. "None" means the function ignores this argument and leaves it to the API to figure out.
    model_state="all",            # Choose from all, known and custom.
    x_fields="name",              # What fields the API should return. If return_raw=True, this is set to "name" by the function, since that's all it's going to return.
    timeout=horde.inf_timeout,    # inf_timeout = None, meaning no timeout. The AI Horde sometimes hangs minutes on simply for sending over model names.
    retries=horde.default_retries # default retries is 3, meaning that if the AI Horde returns malformed or invalid response, it just retries.
)
# Above function can also be run as horde.fetch_models(), since everything you saw,
# except x_fields, that defaults to "performance,queued,jobs,eta,name,count", is
# the default value. However, as noted, return_raw being False is going to set
# x_fields to "name" anyway.

# If out of two tuple arguments, the second is False, it means that the first argument
# only contains a list of exceptions (list[Exception]). If the second arg. if True,
# the function returned the correctly processed response.
if not models[1]:
    horde.log("Failed to fetch models.", 'e') # you could also call print, horde.log just adds fancy formatting. 'n' -> normal, 'e' -> error (with red color), 'u' update (allows refreshing line and adds color)
    horde.log(dumps([str(v) for v in models[0]], indent=2), 'e')
    exit(1)
models = models[0] # escape tuple, now models is simply list[str], where each item is a model name / id

for index, model in enumerate(models):
    horde.log(f"{index}. {model}")
model = models[int(input(">_ Enter desired model's index: "))]

# Fetch model statistics
model_stats: tuple[dict, bool] | tuple[list[Exception], bool] = horde.fetch_model(
    model_name=model, model_type="txt2img",
    x_fields="count,performance,queued,jobs,eta,type,name", # all fields are default
    retries=horde.default_retries,
    timeout=horde.inf_timeout
)
# the function only requires model_name, the rest you're seeing is default.

if not model_stats[1]:
    horde.log(f"Failed to fetch stats for {model}.", 'e')
    horde.log(dumps([str(v) for v in model_stats[0]], indent=2), 'e')
    exit(1)
model_stats = model_stats[0]
horde.log(f"Performance statistics for {model}:\n{dumps(model_stats, indent=2)}")

# Until here, this file is a 1:1 copy-paste of example_txt2txt.py,
# with the only exception being that setting an api-key is generally
# required for image generation, and "txt2img" instead of "txt2txt" at
# horde.fetch_models(model_type="txt2img", ...) and horde.fetch_model(model_type="txt2img", ...).

# Now we've selected a model we can generate images with.
width: int = 512
height: int = 768
download_image: bool = False # wether the function should return bytes() (for webp) for raw link/b64 (latter if you disabled Cloudflare R2)
# and so on... all parameters that the AI Horde supports are also supported by this module.

while True:
    prompt: str = str(input(">_ \033[38;2;192;255;192m")) + ", vivid, highest quality, beautiful, aesthetic"
    print("\033[0m", end='')
    print("\033[38;2;128;128;128mGenerating...\033[0m", end='\r') # Just prints Generating... in grey
    negative_prompt: str = "lowres, low quality, polydactyly, ugly, grotesque, hideous, unattractive, unsightly, displeasing"
    
    timer_start: float = time()
    completion: tuple[dict, bool] | tuple[list[Exception] | bool] = horde.generate_txt2img_completion(
        models=[model], # supports multiple models, we're only using one for ease of input
        width=width,
        height=height,
        prompt=f"{prompt}###{negative_prompt}", # as required by the AI Horde API, negative prompts are to be added as second args, following three hashtags.
        download_image=download_image,
        do_log=True # shows queue position, kudos consumed, ETA, ...
    )
    total_time = round(time() - timer_start, 2)

    if not completion[1]:
        horde.log("Failed to generate image.", 'e')
        horde.log(dumps([str(v) for v in completion[0]], indent=2), 'e')
        if not isinstance(completion[-1], bool):
            horde.log(f"URL that returned error: {completion[-1]}")
            # completions[2] would work too
            # if possible, the function returns the specific URL
            # that made it fail, commonly relevant to the exceptions catched.
    else:
        completion = completion[0] # has keys kudos (int) and generations (list[dict])
        # kudos: how many kudos were consumed to fulfill your request
        # generations: a list if dicts that each contain:
        # - img:
        #  - If you set download_image to True, the raw bytes you can directly safe using open("img.webp", "wb").write(completion["generations"][0]["img"])
        #  - If you set download_image to False and R2 to False, the base64-encoded image (webp format)
        #  - If you set download_image to False (default) and R2 to True (default), a direct link that hosts the webp image
        # - seed: seed used, AI Horde uses a string for some reason
        # - id: request id
        # - censored: if the image is censored
        # - gen_metadata: other stuff you might want to know, in a list
        # - worker_id: worker's ID
        # - worker_name: user's name who's computer fulfilled your request, mapped to worker_id
        # - model: model name that was used to fulfill your request, we specified this using models=[model]
        # - state: is generally "ok", I haven't seen another value.

        if download_image:
            try:
                open("img.webp", "wb").write(completion["generations"][0]["img"])
                horde.log("Saved image to img.webp.")
            except:
                horde.log("Failed to save image to img.webp.", 'e')
        else:
            horde.log(f"To see your completion, access it through the R2 Cloudflare link below.\n{completion['generations'][0]['img']}")

        # Using message type 'u' and adding a newline at the end to make the message colorful, no other reason. Looks better imo.
        horde.log(f"Your request was fulfilled by {completion['generations'][0]['worker_name']} ({completion['generations'][0]['worker_id']}) for {completion['kudos']} kudos in {total_time}s.\n", 'u')