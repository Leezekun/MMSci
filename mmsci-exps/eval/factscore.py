import os
import re
import time
import random
import json
import string
import numpy as np
from openai import AzureOpenAI
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


model = client = OpenAI(api_key="OPENAI_API_KEY")

def extract_float(s):
    # Regular expression to find float numbers between 0 and 1, including 0 and 1
    pattern = r"0\.\d+|1(?:\.0+)?|0(?:\.0+)?"
    matches = re.findall(pattern, s)

    # Prioritize float numbers first
    for match in matches:
        if "." in match:
            return float(match)

    # If no float is found, return 0 or 1
    if matches:
        return float(matches[0])

    return None


def extract_int(s, scales=[0, 5]):
    # Regular expression to find float numbers between 0 and 1, including 0 and 1
    for score in range(scales[0], scales[1] + 1):
        if str(score) in s:
            return int(score)
    return None


def generate_atomic_facts(generated_caption):
    global model

    messages = [
        {
            "role": "system",
            "content": "You will be given a caption for a figure containing multiple panels, which includes descriptions for the entire figure as well as each individual panel. Your task is to break down the caption into separate, independent descriptions for the entire figure and each panel, formatted appropriately and separated by '-'",
        },
        {
            "role": "user",
            "content": """The figure consists of four sub-figures labeled a, b, c, and d. All four images appear to be scanning electron microscope (SEM) images showing the microstructure of different materials, likely related to the iron-based cathode catalysts described in the article.\n\na. This image shows a highly porous structure with interconnected particles forming a network. The particles appear to be in the nanometer to micrometer size range. The scale bar indicates 1 \u03bcm.\n\nb. This image displays a closer view of what seems to be a similar material to (a), but at a higher magnification. The individual particles are more clearly visible, showing a rough, granular texture. The scale bar indicates 200 nm.\n\nc. This image reveals a different morphology compared to (a) and (b). It shows larger, more consolidated structures with a rougher surface texture. There are still visible pores and gaps between the structures. The scale bar indicates 1 \u03bcm.\n\nd. This final image appears to be a cross-sectional view of a porous material, possibly showing the internal structure of the catalyst. It reveals a highly interconnected network of pores and channels throughout the material. The scale bar indicates 200 nm.\n\nThese images likely represent different stages or variations of the iron-acetate/phenanthroline/zeolitic-imidazolate-framework-derived electrocatalyst mentioned in the article. The varying structures and porosities shown in these images could be related to the enhanced mass-transport properties and increased volumetric activity described in the text.""",
        },
        {
            "role": "assistant",
            "content": """
- The figure consists of four sub-figures labeled a, b, c, and d.
- All four images appear to be scanning electron microscope (SEM) images.
- The images show the microstructure of different materials.
- The materials are likely related to the iron-based cathode catalysts described in the article.
- Image a shows a highly porous structure with interconnected particles forming a network.
- The particles in image a are in the nanometer to micrometer size range.
- The scale bar in image a indicates 1 μm.
- Image b displays a closer view of a material similar to that in image a but at higher magnification.
- The individual particles in image b are more clearly visible and show a rough, granular texture.
- The scale bar in image b indicates 200 nm.
- Image c shows larger, more consolidated structures with a rougher surface texture.
- There are visible pores and gaps between the structures in image c.
- The scale bar in image c indicates 1 μm.
- Image d appears to be a cross-sectional view of a porous material.
- Image d reveals the internal structure of the catalyst with a highly interconnected network of pores and channels.
- The scale bar in image d indicates 200 nm.
- These images likely represent different stages or variations of the iron-acetate/phenanthroline/zeolitic-imidazolate-framework-derived electrocatalyst mentioned in the article.
- The varying structures and porosities shown in these images could be related to the enhanced mass-transport properties described in the text.
- The varying structures and porosities in the images may contribute to increased volumetric activity described in the article.
""",
        },
        {"role": "user", "content": f"{generated_caption}"},
    ]

    outputs = model.chat.completions.create(
        model="gpt-4o", messages=messages, temperature=0.7
    )

    print(outputs.choices[0].message.content)

    return outputs.choices[0].message.content


def refine_atomic_facts(generated_facts):
    global model

    format_ex = """
- McCoy played for the Broncos.
- McCoy played for the Broncos during his professional career.
- McCoy played for the Minnesota Vikings.
- McCoy played for the Minnesota Vikings during his professional career.
- McCoy played for the Jacksonville Jaguars."""
    messages = [
        {"role": "assistant", "content": f"{format_ex}"},
        {
            "role": "system",
            "content": "Please revise and strictly itemize the input with '- ' as the example format above if needed. Please generate the itemized examples only without any additional output and erase all the headings.",
        },
        {"role": "user", "content": f"{generated_facts}"},
    ]

    outputs = model.chat.completions.create(
        model="gpt-4o", messages=messages, temperature=0.7
    )

    print(outputs.choices[0].message.content)

    return outputs.choices[0].message.content


def get_score(
    topic,
    abstract,
    content,
    gamma=10,
    scales=[0, 5],
    atomic_facts=None,
):
    examples = [
        {
            "role": "user",
            "content": """
        
        """,
        },
        {"role": "assistant", "content": ""},
    ]
    total_decisions = []
    scores = []
    init_scores = []

    decisions = []
    total_words = 0
    for atom in atomic_facts:
        if not atom:
            continue
        atom = atom.strip()

        definition = f"How relevant is the generated caption to the given human-written caption for the figure? Determine whether the information in the generated caption is included or mentioned in the human-written caption. Respond with a score between {scales[0]} and {scales[1]}."

        prompt = "Human-written caption: {}\n\nGenerated caption: {}\n\nHow relevant is the generated caption to the given human-written caption? Respond with ONLY a score between {} and {}.".format(
            content, atom, scales[0], scales[1]
        )

        messages = [
            {"role": "system", "content": definition},
            {"role": "user", "content": prompt},
        ]

        outputs = model.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0.7
        )

        generated_answer = outputs.choices[0].message.content.lower().strip()

        # generated_score = extract_float(generated_answer)
        generated_score = extract_int(generated_answer, scales=scales)

        if generated_score is None:
            generated_score = 0.0
        else:
            try:
                # normalize the score to 0-1
                generated_score = float(generated_score) / scales[1]
            except Exception as e:
                print("Error:", e)
                generated_score = 0.0

        print("Atom score:", generated_score)

        # if "true" in generated_answer or "false" in generated_answer:
        #     if "true" in generated_answer and "false" not in generated_answer:
        #         is_supported = True
        #     elif "false" in generated_answer and "true" not in generated_answer:
        #         is_supported = False
        #     else:
        #         is_supported = generated_answer.index("true") > generated_answer.index(
        #             "false"
        #         )
        # else:
        #     is_supported = all(
        #         [
        #             keyword
        #             not in generated_answer.lower()
        #             .translate(str.maketrans("", "", string.punctuation))
        #             .split()
        #             for keyword in ["not", "cannot", "unknown", "information"]
        #         ]
        #     )

        decisions.append({"atom": atom, "is_supported": generated_score})

    score = np.mean([d["is_supported"] for d in decisions])
    if gamma:
        init_scores.append(score)
        penalty = (
            1.0 if len(atomic_facts) > gamma else np.exp(1 - gamma / len(atomic_facts))
        )
        score = penalty * score

    total_decisions.append(decisions)
    scores.append(score)
    print("Final score:", score)

    out = {
        "score": np.mean(scores),
        "decisions": total_decisions,
        "num_facts_per_response": np.mean(
            [len(d) for d in total_decisions if d is not None]
        ),
    }

    # print(json.dumps(out,indent=4))

    if gamma:
        out["init_score"] = np.mean(init_scores)

    return out


def iterate_json_file(file_path, num_samples=200, generate=True):
    # Get the answer
    answer_json = "/home/ubuntu/MMSci/mmsci-data/benchmark/test/image_caption_generation_data_w_answer.json"
    with open(answer_json) as f:
        answers = json.load(f)
    answer_dict = {dp["abstract"][:20]: dp["caption"] for dp in answers}

    with open(file_path, "r", encoding="utf-8") as file:
        total_data = json.load(file)

    output_file = file_path.replace(".json", "_factscore6.json")
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

            if "factscore" in data or "prediction" not in data:
                continue

            generated_caption = random.choice(data["prediction"])
            generated_facts = generate_atomic_facts(generated_caption)
            # generated_facts = refine_atomic_facts(generated_facts)

            formatted_facts = generated_facts.split("-")
            topic = data["subject"]
            abstract = data["abstract"]

            # ground-truth caption
            content = data["caption"]
            if not content:
                content = answer_dict[abstract[:20]]
            assert content, f"Caption not found for {abstract[:20]}"

            score_out = get_score(
                topic,
                abstract,
                content,
                gamma=10,
                scales=[0, 5],
                atomic_facts=formatted_facts,
            )

            time_for_single_data = round(time.time() - start_time, 2)

            print("Used time for single data:", time_for_single_data)

            annotated_total_data[idx]["factscore"] = score_out["score"]
            annotated_total_data[idx]["num_facts"] = score_out["num_facts_per_response"]

            with open(output_file, "w") as new_file:
                json.dump(annotated_total_data, new_file, indent=4)

            # print(dp["caption"], "\n", data["prediction"], "\n", data["factscore"], "\n", data["num_facts"])
            # _ = input("Press ENTER to continue......")

    # Summarize
    count, total_score, total_facts = 0, 0.0, 0
    for dp in annotated_total_data:
        if "factscore" in dp:
            count += 1
            total_score += dp["factscore"]
            total_facts += dp["num_facts"]

    average_score = total_score / count if count > 0 else 0
    average_facts = total_facts / count if count > 0 else 0
    final_result = {
        "filepath": file_path,
        "count": count,
        "average_score": average_score,
        "average_facts": average_facts,
    }
    print(
        file_path,
        f"Total {count} files",
        f"average score: {average_score}",
        f"average facts: {average_facts}",
    )
    return final_result


def process_model(
    model_name, num_samples=200, abstract=True, context=False, generate=False
):
    for k in [1, 3, 5]:
        file_name = f"./output/image_caption_generation/abs{abstract}_ctx{context}/k_{k}/{model_name}.json"
        if os.path.exists(file_name):
            break

    if not os.path.exists(file_name):
        print("File not found:", file_name)
        return

    final_result = iterate_json_file(file_name, num_samples, generate)
    output_path = f"./eval_scores/image_caption_generation/factscore-v6/abs{abstract}_ctx{context}/k_{k}/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = f"{output_path}{model_name}.json"
    with open(output_file, "w") as file:
        json.dump(final_result, file, indent=4)


if __name__ == "__main__":
    model_names = [
        "gpt-4o",
        "gpt-4-turbo",
        "gemini-1.5-pro-001",
        "gemini-1.5-pro-002",
        "claude-3-5-sonnet-20240620",
        "qwen2-vl-2b-mmsci-mixed",
        "qwen2-vl-2b",
        "qwen2-vl-7b",
        "qwen",
        "llava",
        "llava-next-mistral",
        "llava-next",
        "blip2",
        "kosmos2",
        "minicpm",
        "idefics2-8b",
        "idefics3-8b",
        "llama3.2-11b",
        "internvl2-1b",
        "internvl2-2b",
        "internvl2-8b"
    ]

    generate = True
    num_samples = 200

    for abstract in [True, False]:
        for context in [False]:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(
                        process_model,
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
