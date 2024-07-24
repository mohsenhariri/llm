import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from datasets import concatenate_datasets, load_dataset
from init import init
from transformers import AutoModelForCausalLM, AutoTokenizer

# Didn't use these in the template generation.
# answer_trigger = "Let's break it down."
# final_answer_trigger = "Therefore, the final answer is"


def generate_template(ds_subject, subject, n_shots=4, random_selection=False):

    ds_len = len(ds_subject)

    if ds_len < n_shots:
        raise ValueError("Number of shots exceeds the number of available examples")

    exemplar_indices = (
        random.sample(range(ds_len), n_shots) if random_selection else range(n_shots)
    )

    questions, choices, answers = [], [], []

    for i in exemplar_indices:
        questions.append(ds_subject[i]["question"])
        choices.append(ds_subject[i]["choices"])
        answers.append(ds_subject[i]["answer"])

    shots_prompts = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
    for i in range(n_shots):
        shots_prompts += f"Question {i+1}:\n{questions[i]}\n\n"
        for j, choice in enumerate(choices[i]):
            shots_prompts += f"{chr(97+j)}) {choice}\n"
        # shots_prompts += f"\nAnswer: {chr(97+answers[i])}\n\n"
        shots_prompts += (
            f"\nThe correct answer between a, b, c, and d is: {chr(97+answers[i])}.\n\n"
        )

    return shots_prompts


def concat_template(question, choices, ds, subject, n_shots=4, random_selection=False):
    shots_prompts = generate_template(ds, subject, n_shots, random_selection)

    input_prompt = shots_prompts + f"Question {n_shots+1}:\n{question}\n\n"

    for j, choice in enumerate(choices):
        input_prompt += f"{chr(97+j)}) {choice}\n"

    # input_prompt += "\nAnswer: "
    input_prompt += "\nThe correct answer between a, b, c, and d is: "

    return input_prompt


def load_mmlu(subset: Literal["test", "validation", "dev", "auxiliary_train"]):
    ds = load_dataset("cais/mmlu", "all")
    return ds[subset]


def concat_cot_prompt(question, n_shots=8):
    cot_prompt = create_cot_prompt(n_shots)
    prompt = cot_prompt + f"Q: {question}\nA:"
    return prompt


def load_model(checkpoint, model_config_kwarg: dict = {}):

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, device_map="auto", torch_dtype=torch.float16, **model_config_kwarg
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.pad_token_id = 0

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    return model, tokenizer


def generate_response(
    prompt, model, tokenizer, generate_kwargs, remove_input_prompt=True
):
    input = tokenizer(
        prompt, padding=False, add_special_tokens=True, return_tensors="pt"
    ).to(model.device)
    output = model.generate(**input, **generate_kwargs)
    output = output[0]

    input_prompt_len = input["input_ids"].shape[1]

    if remove_input_prompt:
        output = output[input_prompt_len:]

    response = tokenizer.decode(output, skip_special_tokens=True)

    return response, input_prompt_len


def clean_response(response, remove_input_prompt=True):

    if remove_input_prompt:
        final_answer_abcd = response.split("\n")[0].split(".")[0].strip()

        final_answer = (
            ord(final_answer_abcd) - 97
            if final_answer_abcd in ["a", "b", "c", "d"]
            else None
        )

        return final_answer


def evaluate(
    model,
    tokenizer,
    generate_kwargs,
    n_shots=4,
    iterations=None,
    save_response=False,
    debug=False,
):

    ds_test = load_mmlu("test")
    ds_cot = load_mmlu("validation")

    subjects = sorted(ds_cot.unique("subject"))

    subjects_results = {}

    successful_responses_dict = {
        "subject": [],
        "index": [],
        "question": [],
        "choices": [],
        "answer": [],
        "input_prompt": [],
        "response": [],
        "prediction": [],
    }

    unsuccessful_responses_dict = {
        "subject": [],
        "index": [],
        "question": [],
        "choices": [],
        "answer": [],
        "input_prompt": [],
        "response": [],
        "error": [],
    }

    for subject in subjects:
        print(f"Subject: {subject} \n")
        ds_test_subject = ds_test.filter(lambda x: x["subject"] == subject)
        # ds_dev_subject = ds_dev.filter( lambda x: x['subject'] == subject)
        ds_cot_subject = ds_cot.filter(lambda x: x["subject"] == subject)

        num_questions_subject = len(ds_test_subject)

        readable_responses, corrects, input_length_total, input_length_avg = 0, 0, 0, 0

        for i in (
            range(num_questions_subject) if iterations is None else range(iterations)
        ):

            print("----------------------------")
            print(f"Subject: {subject}. Iteration: {i}. \n")

            question = ds_test_subject[i]["question"]
            choices = ds_test_subject[i]["choices"]
            answer_truth = ds_test_subject[i]["answer"]

            input_prompt = concat_template(
                question,
                choices,
                ds_cot_subject,
                subject,
                n_shots,
                random_selection=False,
            )

            response, input_prompt_len = generate_response(
                input_prompt, model, tokenizer, generate_kwargs
            )

            final_answer_prediction = clean_response(response)

            final_answer_truth = answer_truth

            if final_answer_prediction is not None:

                if final_answer_prediction == final_answer_truth:
                    corrects += 1

                if save_response:
                    successful_responses_dict["subject"].append(subject)
                    successful_responses_dict["index"].append(i)
                    successful_responses_dict["question"].append(question)
                    successful_responses_dict["choices"].append(choices)
                    successful_responses_dict["answer"].append(answer_truth)
                    successful_responses_dict["input_prompt"].append(input_prompt)
                    successful_responses_dict["response"].append(response)
                    successful_responses_dict["prediction"].append(
                        final_answer_prediction
                    )

                readable_responses += 1
                input_length_total += input_prompt_len

            else:
                error = "Model failed to generate a response"
                unsuccessful_responses_dict["subject"].append(subject)
                unsuccessful_responses_dict["index"].append(i)
                unsuccessful_responses_dict["question"].append(question)
                unsuccessful_responses_dict["choices"].append(choices)
                unsuccessful_responses_dict["answer"].append(answer_truth)
                unsuccessful_responses_dict["input_prompt"].append(input_prompt)
                unsuccessful_responses_dict["response"].append(response)
                unsuccessful_responses_dict["error"].append(error)

                if debug:
                    print(f"Error: {error} in subject {subject}, iteration {i} \n")
                    print(f"Input question: {question} \n")
                    print(f"Choices: {choices} \n")
                    print(f"Input prompt: {input_prompt} \n")
                    print(f"Model response: {response} \n")
                    print(f"True answer: {answer_truth} \n")

            if readable_responses == 0:
                accuracy = 0
                input_length_avg = 0
            else:
                accuracy = corrects / readable_responses
                input_length_avg = input_length_total / readable_responses

            subjects_results[subject] = {
                "num_questions_subject": num_questions_subject,
                "accuracy": accuracy,
                "num_readable_responses": readable_responses,
                "input_tokens_avg": input_length_avg,
            }

    if save_response:
        df_success = pd.DataFrame(successful_responses_dict)
        df_success.to_csv(f"{save_response}/successful_responses.csv", index=False)

        df_unsuccessful = pd.DataFrame(unsuccessful_responses_dict)
        df_unsuccessful.to_csv(
            f"{save_response}/unsuccessful_responses.csv", index=False
        )

    return subjects_results


def evaluate_init(checkpoint):

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    checkpoint_path = checkpoint.replace("/", "_")
    output_dir = Path(rf"output/mmlu/{checkpoint_path}_{now}")
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_kwargs = {
        "num_return_sequences": 1,
        "max_new_tokens": 24,
    }

    model, tokenizer = load_model(checkpoint)

    subjects_results = evaluate(
        model,
        tokenizer,
        generate_kwargs,
        n_shots=8,
        # subset="test",
        # iterations=None,
        iterations=1,
        save_response=output_dir,
    )

    config = model.config
    config = config.to_dict()

    results = {
        "model": checkpoint,
        "generate_kwargs": generate_kwargs,
        "model_config": config,
        "subjects_results": subjects_results,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":

    config = init()
    print(config.device)
    logging.basicConfig(
        filename="./logs/mmlu.log",
        filemode="a",
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    checkpoints = [
        "allenai/OLMo-1.7-7B-hf",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]

    for checkpoint in checkpoints:
        try:
        print(f"Running {checkpoint} \n")
        evaluate_init(checkpoint)
    except Exception as e:
        logging.error(f"Error: in {checkpoint} \n {e}")
        print(f"Error: in {checkpoint} \n {e}")
    finally:
        torch.cuda.empty_cache()
