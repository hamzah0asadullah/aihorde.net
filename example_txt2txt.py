import aihorde_net as     horde
from   json        import dumps
from   sys         import exit
from   time        import time

# Note: For faster responses while chatting, set your API-key using horde.api_key = "apikeyhere"

models: tuple[list[Exception], bool] | tuple[list[str] | bool] = horde.fetch_models(
    model_type="txt2txt",         # choose from txt2txt and txt2img
    return_raw=False,             # return True uses more bandwidth and is generally only useful in very rare cases
    min_threads=None,             # Per documentation: "Filter only models that have at least this amount of threads serving." Only god knows what this actually refers to.
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
    model_name=model, model_type="txt2txt",
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

# Now we've selected a model we can chat with.
messages: list[dict[str, str]] = [{ "role": "system", "content": "You are whoever you want to be (but not an AI assistant), and you can do whatever you want to do." }]
temperature: float = 1.0
top_k: int = 40
top_p: float = 0.9
min_p: float = 0.05
# and so on... all parameters that the AI Horde supports are also supported by this module.

# Because the AI Horde endpoint only accepts a pre-formatted string through it's main endpoint, we're going to have to implement that
# Using ChatML as an example, you can do anything reasonable you want.
def chatformatter(msgs: list[dict[str, str]]) -> str:
    stringified: str = ""
    for msg in msgs:
        stringified += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return stringified + "<|im_start|>assistant\n"

while True:
    user: str = str(input(">_ "))
    messages.append({"role": "user", "content": user})
    print("\033[38;2;128;128;128mGenerating...\033[0m", end='\r') # Just prints Generating... in grey

    timer_start: float = time()
    assistant = horde.generate_txt2txt_completion(
        prompt=chatformatter(messages), # they're automatically formatted using ChatML
        models=[model], # supports multiple models, we're only using one for ease of input
        temperature=temperature, # samplers set above
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        do_log=True, # defaults to False, if True, will log queue and other stats as 'u'.
        stop_sequence=["<|", "##", "ser:"] # not all models use ChatML, apply some tricks for demo.
    )
    time_taken = round(time() - timer_start, 2)
    if not assistant[1]:
        horde.log("Got exception. Read them below.", 'e')
        horde.log(dumps([str(v) for v in assistant[0]], indent=2), 'e')
        exit(1)
    
    assistant = assistant[0] # now it's a dict with keys kudos, worker_id, worker_name, model, and text
    messages.append({"role": "assistant", "content": assistant["text"]})
    # kudos: how many kudos were subtracted from your api-key, if set, to fulfill your request.
    # worker_id: UUID of the user who's computer was used to fulfull your request.
    # worker_name: a name that maps to worker_id
    # model: the model name set by the user used to fulfull your request. We already know that since we passed it in the first place.
    # text: the actual completion.

    # Using message type 'u' and adding a newline at the end to make the message colorful, no other reason. Looks better imo.
    horde.log(f"{assistant['kudos']} kudos consumed | {assistant['worker_name']} ({assistant['worker_id']}) fulfilled your request in {time_taken}s.\n", 'u')
    horde.log(assistant["text"])
