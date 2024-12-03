"""LLM Log-Likelihood Scoring for OpenAI GPT models.

Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# LLM Log-Likelihood Scoring for OpenAI GPT models.
# Uses the top-5 log probabilities of the model to score a prediction
# as similar or not to a given ground truth answer.S

import datetime
import json
import logging
import os
import pathlib
from typing import Any, Optional, Tuple, List, Dict

import numpy as np
import openai
import scipy
import time
import random
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


NEGATIVE_INF = -1000.0
logger = logging.getLogger()

_PROMPT = "You are given a question, ground-truth answer, and a candidate answer. Question: <question> \nGround-truth answer: <GT> \nCandidate answer: <answer> \n\
Is the semantic meaning of the ground-truth and candidate answers similar? Answer in one word - Yes or No."
_SUFFIXES_TO_SCORE = [" yes", " yeah"]
_COMPLEMENT_SUFFIXES = [" no"]


def renormalize_score(yes_score: float, no_score: float) -> float:
    """Corrects the score by applying log-softmax normalization."""
    return np.exp(yes_score - scipy.special.logsumexp([no_score, yes_score]))


def _normalize(text: str) -> str:
    """Remove white space and lower case for normalized comparisons."""
    return text.strip().lower()


def score_openai(
    response: Dict[str, Any],  # openai.ChatCompletion
    suffixes: List[str],
    complement_suffixes: List[str],
) -> float:
    """Returns renormalized prob(suffix) based on top-5 logprobs."""
    assert len(response["choices"]) == 1, "More than 1 choice."
    response = response["choices"][0]

    # Sanity checks.
    if "logprobs" not in response:
        raise ValueError("No logprobs found.")
    if "content" not in response["logprobs"]:
        raise ValueError("No content found.")
    if not response["logprobs"]["content"]:
        raise ValueError("Content is empty.")
    if "top_logprobs" not in response["logprobs"]["content"][0]:
        raise ValueError("No top_logprobs found.")

    top_answers_logprobs = response["logprobs"]["content"][0]["top_logprobs"]

    # Identify the suffix and complement_suffix if each of them exist.
    # Additionally, extract the corresponding logprob.
    # -- First, search for the suffix.
    suffix_logprob = NEGATIVE_INF
    complement_logprob = NEGATIVE_INF
    suffix_index = -1
    complement_suffix_index = -1

    normalized_suffixes = [_normalize(suffix) for suffix in suffixes]
    normalized_complement_suffixes = [
        _normalize(complement_suffix) for complement_suffix in complement_suffixes
    ]

    # Iterate over the top-n logprobs to find the suffix and complement_suffix.
    # The logprobs are already sorted in descending order, so we break once we
    # find the match with the highest logprob.
    for i, token_logprob in enumerate(top_answers_logprobs):
        if _normalize(token_logprob["token"]) in normalized_suffixes:
            suffix_logprob = token_logprob["logprob"]
            suffix_index = i
            break

    for i, token_logprob in enumerate(top_answers_logprobs):
        if _normalize(token_logprob["token"]) in normalized_complement_suffixes:
            complement_suffix_index = i
            complement_logprob = token_logprob["logprob"]
            break

    logger.info(
        "Found: Suffix index: %d, complement_suffix_index: %d",
        suffix_index,
        complement_suffix_index,
    )

    # None of the suffixes or complement_suffixes were found in the output.
    # So score is 0.0.
    if suffix_index == -1 and complement_suffix_index == -1:
        return 0.0

    # Both suffix and complement_suffix were found in the output!
    # This indicates model is ambiguous and there's high prob of both.
    if suffix_index != -1 and complement_suffix_index != -1:
        return renormalize_score(yes_score=suffix_logprob, no_score=complement_logprob)

    # If only one of the suffix or complement_suffix was found in the output,
    # then we want to find the logprob of the reciprocal (i.e. the item that was
    # not found). To find the reciprocal we get the
    # max of (lowest top-5 logprobs, remaining log prob after summing the top-5).
    # This is equivalent to identifying the
    # min of (lowest prob token in top-5, remaining prob after summing top-5)
    lowest_logprob = top_answers_logprobs[-1]["logprob"]
    lowest_token_prob = np.exp(lowest_logprob)
    sum_probs = sum(
        [np.exp(token_logprob["logprob"]) for token_logprob in top_answers_logprobs]
    )
    remaining_prob = 1 - sum_probs
    min_prob = min(lowest_token_prob, remaining_prob)
    if min_prob < 1e-8:
        min_prob = 1e-8
    reciprocal_logprob = np.log(min_prob)

    if suffix_index != -1:
        exclude_score = suffix_logprob
        include_score = reciprocal_logprob
    elif complement_suffix_index != -1:
        exclude_score = reciprocal_logprob
        include_score = complement_logprob
    else:
        raise ValueError("Not the case where suffix or complement suffix is found.")

    return renormalize_score(yes_score=exclude_score, no_score=include_score)


class OpenAIClient:
    """A proxy to query a OpenAI's API."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        json_output_path: Optional[str],
    ):
        """Initializes a OpenAIClient.

        Args:
          model_name: The name of the OpenAI model to use (e.g. 'gpt-4').
          api_key: OpenAI API key string.
          json_output_path: If not None, the path to the directory to write JSON
            output to.
        """

        self.client = OpenAI(
            api_key=api_key,
            timeout=1200,
            max_retries=5,
        )

        self._model_name = model_name
        if json_output_path:
            self._json_output_path = pathlib.Path(json_output_path)
            if not self._json_output_path.exists():
                self._json_output_path.mkdir(parents=True, exist_ok=True)
        self._timeout = 60

    def call_openai(
        self,
        prompt: str,
        output_prefix: Optional[str],
        max_decode_steps: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Call OpenAI chat completion API; save and return the response."""
        message = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_decode_steps,
            top_p=1,
            timeout=self._timeout * 10,
        )
        # response = openai.chat.completions.create(
        #     model=self._model_name,
        #     messages=message,
        #     temperature=temperature,
        #     max_tokens=max_decode_steps,
        #     top_p=1,
        #     timeout=self._timeout * 10,
        # )
        response_json = response.model_dump_json()
        response = json.loads(response_json)
        assert len(response["choices"]) == 1
        if not output_prefix:
            output_prefix = ""
        if self._json_output_path:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(
                self._json_output_path, f"gpt4_{output_prefix}_{timestamp}.json"
            )
            with open(filename, "w") as f:
                response["input_prompt"] = message
                json.dump(response, f)
        text_response = response["choices"][0]["message"]["content"]
        return text_response

    def call_openai_with_score(
        self,
        prompt: str,
        suffixes: List[str],
        output_prefix: Optional[str],
        max_decode_steps: int = 1024,
        temperature: float = 0.0,
        complement_suffixes: Optional[List[str]] = None,
    ) -> Tuple[str, float]:
        """Call OpenAI."""
        message = [{"role": "user", "content": prompt}]
        if not output_prefix:
            output_prefix = ""
        assert suffixes, "Please supply a suffix token to score the output."
        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_decode_steps,
            top_p=1,
            logprobs=True,
            top_logprobs=5,  # This is the largest value allowed by OpenAI.
            frequency_penalty=0,
            presence_penalty=0,
            timeout=self._timeout * 10,
        )
        response_json = response.model_dump_json()
        response = json.loads(response_json)
        assert len(response["choices"]) == 1
        if self._json_output_path:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(
                self._json_output_path, f"gpt4_{output_prefix}_{timestamp}.json"
            )
            with open(filename, "w") as f:
                response["input_prompt"] = message
                json.dump(response, f)
        text_response = response["choices"][0]["message"]["content"]
        if complement_suffixes is None:
            complement_suffixes = []
        score = score_openai(
            response, suffixes=suffixes, complement_suffixes=complement_suffixes
        )
        return text_response, score


def iterate_json_file(client, file_path, num_samples=200, generate=True):
    # Get the answer
    answer_json = "/home/ubuntu/MMSci/mmsci-data/benchmark/test/image_caption_generation_data_w_answer.json"
    with open(answer_json) as f:
        answers = json.load(f)
    answer_dict = {dp["abstract"][:20]: dp["caption"] for dp in answers}

    with open(file_path, "r", encoding="utf-8") as file:
        total_data = json.load(file)

    output_file = file_path.replace(".json", "_l3score6.json")
    if os.path.exists(output_file):
        with open(output_file, "r") as file:
            annotated_total_data = json.load(file)
        print("Loaded existing data")
    else:
        annotated_total_data = []

    existing_imgs = [dp["image"] for dp in annotated_total_data]

    # Get a subset with 200 examples
    random.shuffle(total_data)
    for dp in total_data:
        if "prediction" not in dp:
            continue
        if not dp["prediction"]:
            continue
        if dp["image"] in existing_imgs:
            continue

        annotated_total_data.append(dp)
        existing_imgs.append(dp["image"])

        if len(annotated_total_data) >= num_samples:
            break

    print(len(annotated_total_data))

    if generate:
        for idx, data in enumerate(annotated_total_data):
            start_time = time.time()

            if "l3score" in data or "prediction" not in data:
                continue

            candidate_answer = random.choice(data["prediction"])

            # ground-truth caption
            gt = data["caption"]
            topic = data["subject"]
            abstract = data["abstract"]
            question = (
                "Generate the caption for the image in the context of "
                + topic
                + ". Discribe each panel in the image."
            )

            if not gt:
                gt = answer_dict[abstract[:20]]
            assert gt, f"Caption not found for {abstract[:20]}"

            prompt_current = (
                _PROMPT.replace("<question>", question)
                .replace("<GT>", gt)
                .replace("<answer>", candidate_answer)
            )

            response, prob_yes = client.call_openai_with_score(
                prompt=prompt_current,
                suffixes=_SUFFIXES_TO_SCORE,
                complement_suffixes=_COMPLEMENT_SUFFIXES,
                output_prefix="",
            )

            time_for_single_data = round(time.time() - start_time, 2)

            print("Score:", prob_yes, "Used time:", time_for_single_data)

            annotated_total_data[idx]["l3score"] = prob_yes
            annotated_total_data[idx]["response"] = response

            with open(output_file, "w") as new_file:
                json.dump(annotated_total_data, new_file, indent=4)

    # Summarize
    count, total_score = 0, 0.0
    for dp in annotated_total_data:
        if "l3score" in dp:
            count += 1
            total_score += dp["l3score"]
            # total_facts += dp["num_facts"]

    average_score = total_score / count if count > 0 else 0
    # average_facts = total_facts / count if count > 0 else 0
    final_result = {
        "filepath": file_path,
        "count": count,
        "average_score": average_score,
        # "average_facts": average_facts,
    }
    print(
        file_path,
        f"Total {count} files",
        f"average score: {average_score}",
        # f"average facts: {average_facts}",
    )
    return final_result


def process_model(
    client, model_name, num_samples=200, abstract=True, context=False, generate=False
):
    for k in [1, 3, 5]:
        file_name = f"./output/image_caption_generation/abs{abstract}_ctx{context}/k_{k}/{model_name}.json"
        if os.path.exists(file_name):
            break

    if not os.path.exists(file_name):
        print("File not found:", file_name)
        return

    final_result = iterate_json_file(client, file_name, num_samples, generate)
    output_path = f"./eval_scores/image_caption_generation/l3score/abs{abstract}_ctx{context}/k_{k}/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = f"{output_path}{model_name}.json"
    with open(output_file, "w") as file:
        json.dump(final_result, file, indent=4)


if __name__ == "__main__":
    client = OpenAIClient(
        model_name="gpt-4o",
        api_key="YOUR_OPENAI_APIKEY",
        json_output_path="./saved_output_l3score/",
    )

    question = "Where is Niagara falls located?"
    gt = "Niagara Falls is located on the border between the United States and Canada, specifically between New York State and Ontario Province."
    candidate_answer = "Niagara Falls is situated on the Niagara River, which connects Lake Erie to Lake Ontario, \
    and lies on the international border between the United States (New York State) and Canada (Ontario Province)."

    prompt_current = (
        _PROMPT.replace("<question>", question)
        .replace("<GT>", gt)
        .replace("<answer>", candidate_answer)
    )

    response, prob_yes = client.call_openai_with_score(
        prompt=prompt_current,
        suffixes=_SUFFIXES_TO_SCORE,
        complement_suffixes=_COMPLEMENT_SUFFIXES,
        output_prefix="",
    )

    # Evaluated models
    model_names = [
        "gpt-4o",
        "gpt-4-turbo",
        "gemini-1.5-pro-001",
        "gemini-1.5-flash-001",
        "claude-3-5-sonnet-20240620",
        "qwen2-vl-2b-mmsci",
        "qwen2-vl-7b-mmsci",
        "qwen2-vl-2b",
        "qwen2-vl-7b",
        "qwen",
        "llava",
        "llava-next-mistral",
        "llava-next",
        "kosmos2",
        "minicpm",
        "idefics2-8b",
        "idefics3-8b",
        "llama3.2-11b",
        "internvl2-2b",
        "internvl2-8b",
    ]

    generate = True
    num_samples = 200

    for abstract in [False, True]:
        for context in [False]:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(
                        process_model,
                        client,
                        model_name,
                        num_samples,
                        abstract,
                        context,
                        generate,
                    )
                    for model_name in model_names
                ]
                for future in as_completed(futures):
                    try:
                        future.result()  # Handle any exceptions raised during execution
                    except Exception as e:
                        print(f"An error occurred: {e}")
