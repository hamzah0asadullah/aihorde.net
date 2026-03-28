# Standard packages
from json   import loads, dumps
from time   import sleep, time
from os     import get_terminal_size
from random import randint
from base64 import encode, decode, b64decode, b64encode

# External
from requests            import post, get, delete, Response
from requests.exceptions import ReadTimeout

### - ###
# To-Do #
### - ###

"""
- Find a way to merge generate_txt2txt_completion and generate_txt2img_completion
- Implement functions to find out if Horde is under high load or not to determine what timeout is reasonable
- Write a documentation on GitHub that allows users to use functions effectively
"""

### ----------------------- ###
# Client and option variables #
### ----------------------- ###

do_log: bool = True
api_key: str = "0" * 10
base_endpoint: str = "https://aihorde.net/api/v2/"

default_retries: int = 3
quick_timeout:  float = 5.0
medium_timeout: float = 15.0
slow_timeout:   float = 30.0
inf_timeout:    float | None = None

### ------------- ###
# Generic functions #
### ------------- ###

def log(msg: str, msg_type: str = 'n', prefix: str = ">> ", do_log: bool = True) -> None:
    if not do_log:
        return
    msg = msg.replace("\n", f"\n{prefix}")
    buffer: str = " " * (get_terminal_size().columns - len(msg) - 5)
    if msg_type == 'u':
        print(f"{prefix}\033[38;2;{randint(192, 255)};{randint(192, 255)};{randint(192, 255)}m{msg}{buffer}\033[0m", end="\r")
    elif msg_type == 'n':
        print(f"{prefix}{msg}{buffer}")
    elif msg_type == 'e':
        print(f"{prefix}\033[38;2;255;150;120m{msg}{buffer}\033[0m")

def fetch_models(model_type: str = "txt2txt", return_raw: bool = False, min_threads: int | None = None, max_threads: int | None = None, model_state: str = "all", x_fields: str | None = "performance,queued,jobs,eta,name,count", retries: int = default_retries, timeout: float | None = inf_timeout) -> tuple:
    model_types:  list[str] = ["txt2txt", "txt2img"]
    model_states: list[str] = ["known", "custom", "all"]

    if model_type not in model_types:
        return [Exception(f"Passed model_type {model_type} is not recognized. Choose from {dumps(model_types)}.")], False
    if model_state not in model_states:
        return [Exception(f"Passed model_state {model_state} is not recognized. Choose from {dumps(model_states)}.")], False
    if not return_raw:
        x_fields = "name"

    translation_dict: dict[str, str] = { "txt2txt": "text", "txt2img": "image" }
    translated_type: str = translation_dict[model_type]

    exceptions: list[Exception] = []
    models_endpoint: str = f"{base_endpoint}status/models?type={translated_type}&model_state={model_state}{f'&min_count={min_threads}' if min_threads != None else ''}{f'&max_count={max_threads}' if max_threads != None else ''}"
    headers: dict = {"X-Fields": x_fields} if x_fields != None else {}
    for _ in range(retries):
        try:
            response = get(models_endpoint, timeout=timeout)
            response.raise_for_status()
            response = response.json()
            return (response if return_raw else [v["name"] for v in response]), True
        except ReadTimeout as e:
            exceptions.append(Exception(f"Read timed out at specified {timeout}s."))
        except Exception as e:
            exceptions.append(e)
    return exceptions, False

def fetch_model(model_name: str, model_type: str = "txt2txt", x_fields: str | None = "count,performance,queued,jobs,eta,type,name", retries: int = default_retries, timeout: float | None = inf_timeout) -> tuple:
    # Note: I know the AI Horde thinks it has an endpoint for this, but that endpoint cannot handle forward slashes (even if replaced by %2F) while most of its models have that in their name.
    response = fetch_models(model_type=model_type, x_fields=x_fields, timeout=timeout, retries=retries, return_raw=True)
    if not response[1]:
        return response
    for v in response[0]:
        if v["name"] == model_name:
            return v, True
    return [Exception(f"\"{model_name}\" can not be found.")], False

def fetch_news(x_fields: str | list[str] | None = None, retries: int = default_retries, timeout: float | None = inf_timeout) -> tuple:
    # Normally, x_fields would simply be sent to the API, unfortunately the endpoint fails to apply these though.
    if isinstance(x_fields, str):
        x_fields = x_fields.split(',')
        x_fields = [v.strip() for v in x_fields]
    
    exceptions: list[Exception] = []
    news_endpoint: str = f"{base_endpoint}status/news"
    for _ in range(retries):
        try:
            response = get(news_endpoint, timeout=timeout)
            response.raise_for_status()
            response = response.json()

            if isinstance(x_fields, list):
                for i, v in enumerate(response):
                    response[i] = {k: v.get(k) for k in x_fields}

            return response, True
        except ReadTimeout as e:
            exceptions.append(Exception(f"Read timed out at specified {timeout}s."))
        except Exception as e:
            exceptions.append(e)
    return exceptions, False

### ----------------- ###
# TXT2TXT functionality #
### ----------------- ###
def generate_txt2txt_completion(
        payload:               dict       | None = None,
        prompt:                str        | None = None, # Standard OpenAI-format
        frmtadsnsp:            bool       | None = None, # format add spaces, inserts spaces around punctuation
        frmtrmblln:            bool       | None = None, # format remove blank lines, removes empty lines
        frmtrmspch:            bool       | None = None, # format remove speech, removes dialogue-styled markup
        frmttriminc:           bool       | None = None, # format trim incomplete, trimms incomplete fragments.
        rep_pen:               float      | None = None,
        rep_pen_range:         int        | None = None, # > 0
        rep_pen_slope:         float      | None = None, # > 0
        singleline:            bool       | None = None,
        temperature:           float      | None = None, # > 0
        tfs:                   float      | None = None, # TFS -> Tail-Free Sampling, 0 <= x <= 1
        top_a:                 float      | None = None, # > 0
        top_k:                 int        | None = None, # > 0
        top_p:                 float      | None = None, # 0 <= x <= 1
        min_p:                 float      | None = None, # 0 <= x <=
        typical:               float      | None = None, # 0 <= x <= 1
        sampler_order:         list[int]  | None = None, # 0 top k, 1 top a, 2 top p, 3 tfs, 4 typical, 5 temp, 6 rep pen, default 6,0,1,3,4,2,5
        use_default_bad_words: bool       | None = None,
        stop_sequence:         list[str]  | None = None,
        smoothing_factor:      float      | None = None, # > 0
        dynatemp_range:        float      | None = None,
        dynatemp_exponent:     float      | None = None,
        max_contenxt_length:   int        | None = None,
        max_length:            int        | None = None,
        softprompt:            str        | None = None,
        trusted_workers:       bool       | None = None,
        validated_backends:     bool      | None = None,
        slow_workers:          bool       | None = None,
        workers:               list[str]  | None = None,
        worker_blacklist:      bool       | None = None,
        models:                list[str]  | None = None,
        dry_run:               bool       | None = None,
        proxied_account:       str        | None = None,
        extra_source_images:   list[dict] | None = None,
        disable_batching:      bool       | None = None,
        allow_downgrade:       bool       | None = None,
        webhook:               str        | None = None,
        style:                 str        | None = None,
        extra_slow_workers:    bool       | None = None,

        retries: int = default_retries,
        timeout: float | None = inf_timeout,
        response_poll_interval: float = 1.0,
        do_log: bool = False
    ) -> tuple:
    if isinstance(payload, dict):
        missing_arg_exception: str = "When you pass payload manually, payload-related function arguments will be ignored."
    else:
        params: dict[str, str | bool | int | float | list[str] | list[int] | None] = { "frmtadsnsp": frmtadsnsp,  "frmtrmblln": frmtrmblln,  "frmtrmspch": frmtrmspch,  "frmttriminc": frmttriminc, "rep_pen": rep_pen, "rep_pen_range": rep_pen_range, "rep_pen_slope": rep_pen_slope, "singleline": singleline, "temperature": temperature,  "tfs": tfs, "top_a": top_a, "top_k": top_k, "top_p": top_p, "min_p": min_p, "typical": typical, "sampler_order": sampler_order,  "use_default_bad_words": use_default_bad_words, "stop_sequence": stop_sequence, "smoothing_factor": smoothing_factor, "dynatemp_range": dynatemp_range, "dynatemp_exponent": dynatemp_exponent, "max_contenxt_length": max_contenxt_length, "max_length": max_length }
        other: dict[str, str | bool | int | float | list[str] | list[dict] | None] = { "prompt": prompt, "softprompt": softprompt, "trusted_workers": trusted_workers, "validated_backends": validated_backends, "slow_workers": slow_workers, "workers": workers, "worker_blacklist": worker_blacklist, "models": models, "dry_run": dry_run, "proxied_account": proxied_account, "extra_source_images": extra_source_images, "disable_batching": disable_batching, "allow_downgrade": allow_downgrade, "webhook": webhook, "style": style, "extra_slow_workers": extra_slow_workers }
        unused_keys: list[str] = []
        for k, v in params.items():
            if v == None:
                unused_keys.append(k)
        for k in unused_keys:
            del params[k]

        unused_keys = []
        for k, v in other.items():
            if v == None:
                unused_keys.append(k)
        for k in unused_keys:
            del other[k]

        payload = other
        payload["params"] = params
        missing_arg_exception: str = "You may set it over function arguments (models=list[str] and messages=list[dict[str, str]])."

    required_args: list[str] = ["models", "prompt"]
    for arg in required_args:
        if arg not in payload:
            if (arg == "models") and ("workers" in payload) and isinstance(payload["workers"], list) and (len(payload["workers"]) > 0):
                pass
            else:
                return [Exception(f"Required argument {arg} missing. {missing_arg_exception}")], False
    
    # Initiate generation and fetch ID
    completion_endpoint: str = f"{base_endpoint}generate/text/async"
    exceptions: list[Exception] = []
    response: None | Response | dict = None
    for _ in range(retries):
        try:
            response = post(completion_endpoint, json=payload, headers={"apikey": api_key}, timeout=timeout)
            try:
                response = response.json()
            except:
                raise Exception(f"{completion_endpoint} responded with malformed (non-JSON) response.")
            if "id" not in response:
                if "message" in response:
                    raise Exception(response["message"])
                else:
                    response.raise_for_status() # if that doesn't raise, line below will.
                    raise Exception(f"AI Horde responded with malformed request (no id and error message): {dumps(response)}")
            break
        except ReadTimeout as e:
            exceptions.append(Exception(f"Read timed out at specified {timeout}s."))
        except Exception as e:
            exceptions.append(e)
    if len(exceptions) == retries:
        return exceptions, False, completion_endpoint
    
    request_id: str = response["id"]
    total_kudos: int = response["kudos"]
    completion_endpoint = f"{base_endpoint}generate/text/status/{request_id}"
    exceptions = []
    response = None
    for _ in range(retries):
        timer_start: float = time()
        try:
            while True:
                response = get(completion_endpoint, headers={"id": request_id}, timeout=timeout)
                try:
                    response = response.json()
                except:
                    raise Exception(f"AI Horde responded with malformed (non-JSON) response.")
                if "finished" not in response:
                    if "message" in response:
                        raise Exception(response["message"])
                    else:
                        response.raise_for_status() # if that doesn't raise, line below will.
                        raise Exception(f"AI Horde responded with malformed request (no id and error message): {dumps(response)}")
                if response["finished"] == 1:
                    break
                log(f"{response['kudos']}/{total_kudos} kudos consumed | queue pos. {response['queue_position']} | ETA {response['wait_time']}s", 'u', do_log=do_log)
                try:
                    sleep(max(0, response_poll_interval - (time() - timer_start)) if (response["wait_time"] == 0) else float(response["wait_time"]))
                except KeyboardInterrupt:
                    delete(f"{base_endpoint}generate/text/status/{request_id}", headers={"id": request_id})
                    return [Exception("KeyboardInterrupt detected, request gracefully aborted.")], False
        except Exception as e:
            exceptions.append(e)
        if isinstance(response, dict) and ("finished" in response) and (response["finished"] == 1):
            break
    if len(exceptions) == retries:
        return exceptions, False, completion_endpoint
    if response["faulted"] == True:
        return ["AI Horde faulted your request. Make sure not to request CSAM content."], False, completion_endpoint
    
    completion: dict = {}
    completion["kudos"]       = response["kudos"]
    completion["worker_id"]   = response["generations"][0]["worker_id"]
    completion["worker_name"] = response["generations"][0]["worker_name"]
    completion["model"]       = response["generations"][0]["model"]
    completion["text"]        = response["generations"][0]["text"]
    return completion, True

### ----------------- ###
# IMG2TXT functionality #
### ----------------- ###
def generate_txt2img_completion(
        payload:                      dict       | None = None,
        prompt:                       str        | None = None,
        sampler_name:                 str        | None = None,
        cfg_scale:                    float      | None = None,
        denoising_strength:           float      | None = None,
        hires_fix_denoising_strength: float      | None = None,
        height:                       int        | None = None,
        width:                        int        | None = None,
        post_processing:              list[str]  | None = None,
        karras:                       bool       | None = None,
        tiling:                       bool       | None = None,
        hires_fix:                    bool       | None = None,
        clip_skip:                    int        | None = None,
        facefixer_strength:           float      | None = None,
        loras:                        list[dict] | None = None,
        tis:                          list[dict] | None = None, # likely textual inversion
        special:                      dict       | None = None,
        workflow:                     str        | None = None,
        transparent:                  bool       | None = None,
        seed:                         str        | None = None,
        seed_variation:               int        | None = None,
        control_type:                 str        | None = None,
        image_is_control:             bool       | None = None,
        return_control_map:           bool       | None = None,
        extra_texts:                  list[dict] | None = None,
        steps:                        int        | None = None,
        n:                            int        | None = None,
        nsfw:                         bool       | None = None,
        trusted_workers:              bool       | None = None,
        validated_backends:           bool       | None = None,
        slow_workers:                 bool       | None = None,
        extra_slow_workers:           bool       | None = None,
        censor_nsfw:                  bool       | None = None,
        workers:                      list[str]  | None = None,
        worker_blacklist:             bool       | None = None,
        models:                       list[str]  | None = None,
        source_image:                 str        | None = None,
        source_processing:            str        | None = None,
        source_mask:                  str        | None = None,
        extra_source_images:          list[dict] | None = None,
        r2:                           bool       | None = None,
        shared:                       bool       | None = None,
        replacement_filter:           bool       | None = None,
        dry_run:                      bool       | None = None,
        proxied_account:              str        | None = None,
        disable_batching:             bool       | None = None,
        allow_downgrade:              bool       | None = None,
        webhook:                      str        | None = None,
        style:                        str        | None = None,
        
        retries_init: int = default_retries,
        timeout_init: float | None = inf_timeout,
        retries_poll: int = default_retries,
        response_poll_interval: float = 5.0,
        retries_result: int = default_retries,
        timeout_result: float | None = inf_timeout,
        download_image: bool = False,
        do_log: bool = False
    ) -> tuple:
    if isinstance(payload, dict):
        missing_arg_exception: str = "When you pass payload manually, payload-related function arguments will be ignored."
    else:
        params: dict[str, dict | str | list | float | bool | None] = { "sampler_name": sampler_name, "cfg_scale": cfg_scale, "denoising_strength": denoising_strength, "hires_fix_denoising_strength": hires_fix_denoising_strength, "height": height, "width": width, "post_processing": post_processing, "karras": karras, "tiling": tiling, "hires_fix": hires_fix, "clip_skip": clip_skip, "facefixer_strength": facefixer_strength, "loras": loras, "tis": tis, "special": special, "workflow": workflow, "transparent": transparent, "seed": seed, "seed_variation": seed_variation, "control_type": control_type, "image_is_control": image_is_control, "return_control_map": return_control_map, "extra_texts": extra_texts, "steps": steps, "n": n }
        other: dict[str, dict | str | list | float | bool | None] = { "prompt": prompt, "nsfw": nsfw, "trusted_workers": trusted_workers, "validated_backends": validated_backends, "slow_workers": slow_workers, "extra_slow_workers": extra_slow_workers, "censor_nsfw": censor_nsfw, "workers": workers, "worker_blacklist": worker_blacklist, "models": models, "source_image": source_image, "source_processing": source_processing, "source_mask": source_mask, "extra_source_images": extra_source_images, "r2": r2, "shared": shared, "replacement_filter": replacement_filter, "dry_run": dry_run, "proxied_account": proxied_account, "disable_batching": disable_batching, "allow_downgrade": allow_downgrade, "webhook": webhook, "style": style }
        unused_keys: list[str] = []

        for k, v in params.items():
            if v == None:
                unused_keys.append(k)
        for k in unused_keys:
            del params[k]

        unused_keys = []
        for k, v in other.items():
            if v == None:
                unused_keys.append(k)
        for k in unused_keys:
            del other[k]

        payload = other
        payload["params"] = params
        missing_arg_exception: str = "You may set it over function arguments (models=list[str] and messages=list[dict[str, str]])."
    
    required_args: list[str] = ["models", "prompt"]
    for arg in required_args:
        if arg not in payload:
            if (arg == "models") and ("workers" in payload) and isinstance(payload["workers"], list) and (len(payload["workers"]) > 0):
                pass
            else:
                return [Exception(f"Required argument {arg} missing. {missing_arg_exception}")], False
    
    valid_samplers: list[str] = ['k_heun', 'k_dpm_adaptive', 'k_dpm_2_a', 'k_euler', 'k_dpm_2', 'k_dpmpp_2m', 'k_euler_a', 'k_dpm_fast', 'k_lms', 'DDIM', 'k_dpmpp_sde', 'lcm', 'k_dpmpp_2s_a', 'dpmsolver']
    if isinstance(sampler_name, str) and (sampler_name not in valid_samplers):
        return [Exception(f"{sampler_name} is not a by the AI Horde supported sampler. Pick from {dumps(valid_samplers)}.")], False

    # Except completion_endpoint, below two codeblocks are just copy-pasted from txt2txt
    completion_endpoint: str = f"{base_endpoint}generate/async"
    exceptions: list[Exception] = []
    response: None | Response | dict = None
    for _ in range(retries_init):
        try:
            response = post(completion_endpoint, json=payload, headers={"apikey": api_key}, timeout=timeout_init)
            try:
                response = response.json()
            except:
                raise Exception(f"{completion_endpoint} responded with malformed (non-JSON) response.")
            if "id" not in response:
                if "message" in response:
                    raise Exception(response["message"])
                else:
                    response.raise_for_status() # if that doesn't raise, line below will.
                    raise Exception(f"AI Horde responded with malformed request (no id and error message): {dumps(response)}")
            break
        except ReadTimeout as e:
            exceptions.append(Exception(f"Read timed out at specified {timeout_init}s."))
        except Exception as e:
            exceptions.append(e)
    if (not isinstance(response, dict)) or ("id" not in response):
        return exceptions, False, completion_endpoint
    
    request_id: str = response["id"]
    total_kudos: int = response["kudos"]
    completion_endpoint = f"{base_endpoint}generate/check/{request_id}"
    exceptions = []
    response = None
    for _ in range(retries_poll):
        try:
            while True:
                timer_start: float = time()
                response = get(completion_endpoint, headers={"id": request_id}, timeout=timeout_init)
                try:
                    response = response.json()
                except:
                    raise Exception(f"AI Horde responded with malformed (non-JSON) response.")
                if "finished" not in response:
                    if "message" in response:
                        raise Exception(response["message"])
                    else:
                        response.raise_for_status() # if that doesn't raise, line below will.
                        raise Exception(f"AI Horde responded with malformed request (no id and error message): {dumps(response)}")
                if response["finished"] == 1:
                    break
                log(f"{response['kudos']}/{total_kudos} kudos consumed | queue pos. {response['queue_position']} | ETA {response['wait_time']}s", 'u', do_log=do_log)
                try:
                    sleep(max(0, response_poll_interval - (time() - timer_start)) if (response["wait_time"] == 0) else float(response["wait_time"]))
                except KeyboardInterrupt:
                    delete(f"{base_endpoint}generate/status/{request_id}", headers={"id": request_id})
                    return [Exception("KeyboardInterrupt detected, request gracefully aborted.")], False
        except Exception as e:
            exceptions.append(e)
        if isinstance(response, dict) and ("finished" in response) and (response["finished"] == 1):
            break
    if len(exceptions) == retries_poll:
        return exceptions, False, completion_endpoint
    if response["faulted"] == True:
        return ["AI Horde faulted your request. Make sure not to request CSAM content."], False, completion_endpoint
    
    completion_endpoint = f"{base_endpoint}generate/status/{request_id}"
    exceptions = []
    response = None
    for _ in range(retries_result):
        try:
            response = get(completion_endpoint, timeout=timeout_result, headers={"id": request_id})
            try:
                response = response.json()
            except:
                raise Exception(f"AI Horde responded with malformed (non-JSON) response.")
            if "message" in response:
                raise Exception(response["message"])
            response.raise_for_status()
            break
        except Exception as e:
            exceptions.append(e)

    completion: dict = {}
    completion["kudos"]       = response["kudos"]
    completion["generations"] = response["generations"]
    exceptions = []
    if download_image:
        if (r2 == True) or (r2 == None):
            for x in range(len(completion["generations"])):
                for _ in range(retries_result):
                    try:
                        img = get(completion["generations"][x]["img"])
                        img.raise_for_status()
                        completion["generations"][x]["img"] = img.content
                        break
                    except Exception as e:
                        exceptions.append(e)
                if retries_result == len(exceptions):
                    return exceptions, False
                else:
                    exceptions = []
        else:
            for x in range(len(completion["generations"])):
                completion["generations"][x]["img"] = b64decode(completion["generations"][x]["img"])
    return completion, True