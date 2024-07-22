import json
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from datasets import concatenate_datasets, load_dataset
from init import init
from transformers import AutoModelForCausalLM, AutoTokenizer

answer_trigger = "Let's break it down."
final_answer_trigger = "Therefore, the final answer is"


def generate_template(question, chain, answer):
    question.append(
        "There are 15 trees in the grove. "
        "Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. "
        "How many trees did the grove workers plant today?"
    )
    chain.append(
        "There are 15 trees originally. "
        "Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6."
    )
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?"
    )
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?"
    )
    chain.append(
        "Originally, Leah had 32 chocolates. "
        "Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39."
    )
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?"
    )
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8."
    )
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?"
    )
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9."
    )
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?"
    )
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29."
    )
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?"
    )
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls."
    )
    answer.append("33")

    question.append(
        "Olivia has $23. She bought five bagels for $3 each. "
        "How much money does she have left?"
    )
    chain.append(
        "Olivia had 23 dollars. "
        "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8."
    )
    answer.append("8")

    return question, chain, answer


def create_cot_prompt(n_shots=8):

    assert n_shots <= 8, "n_shot should be less than or equal to 8"

    questions, chains, answers = [], [], []

    questions, chains, answers = generate_template(questions, chains, answers)

    questions, chains, answers = (
        questions[:n_shots],
        chains[:n_shots],
        answers[:n_shots],
    )

    cot_prompt = ""
    for i in range(n_shots):
        cot_prompt += f"Q: {questions[i]}\nA: {answer_trigger} {chains[i]} {final_answer_trigger} {answers[i]}.\n\n"

    return cot_prompt


def load_gsm8k(subset="test"):
    gsm = load_dataset("openai/gsm8k", "main")

    if subset == "train":
        return gsm["train"]
    elif subset == "test":
        return gsm["test"]
    elif subset == "both":
        gsm_eval = concatenate_datasets([gsm["train"], gsm["test"]])
        gsm_eval = gsm_eval.shuffle(seed=42)
        return gsm_eval


def concat_cot_prompt(question, n_shots=8):
    cot_prompt = create_cot_prompt(n_shots)
    prompt = cot_prompt + f"Q: {question}\nA:"
    return prompt


def load_model(checkpoint):

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # for key in tokenizer.special_tokens_map:
    #     print(f"{key}: {tokenizer.special_tokens_map[key]}")
    #     print(f"ID: {tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map[key])}")

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.pad_token_id = 0

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

    if remove_input_prompt:
        input_prompt_len = input["input_ids"].shape[1]
        output = output[input_prompt_len:]

    response = tokenizer.decode(output, skip_special_tokens=True)

    return response


def clean_response(response, remove_input_prompt=True):

    if remove_input_prompt:
        first_response = response.split(final_answer_trigger)

        if len(first_response) < 2:
            return f"Error: {final_answer_trigger} not found in response"

        first_response = first_response[1].split("\n")[0].strip()[:-1]

        reg_rum_pattern = r"(\d+(\.\d+)?)"

        match = re.search(reg_rum_pattern, first_response)

        if match:
            num_str = match.group(1)
            num = float(num_str) if "." in num_str else int(num_str)
            return num
        else:
            return f"Error: Number not found in response"


def evaluate(
    model,
    tokenizer,
    generate_kwargs,
    n_shots=8,
    subset="test",
    iterations=None,
    save_response=False,
):

    readable_responses, corrects = 0, 0

    gsm_eval = load_gsm8k(subset=subset)
    num_questions = len(gsm_eval)

    output_dict = {"index": [], "question": [], "answer": [], "prediction": []}

    unsuccessful_responses_dict = {
        "index": [],
        "input_prompt": [],
        "response": [],
        "error": [],
    }

    for i in range(num_questions) if iterations is None else range(iterations):

        question, answer_truth = gsm_eval[i]["question"], gsm_eval[i]["answer"]
        answer_truth = answer_truth.split(" ")[-1]

        prompt = concat_cot_prompt(question, n_shots)

        print(f"Iteration, {i} \n")

        # print(f"Inference prompt: {prompt} \n")

        response = generate_response(prompt, model, tokenizer, generate_kwargs)

        # print(f"Response: {response} \n")

        final_answer_prediction = clean_response(response)

        # print(isinstance(final_answer_prediction, str))

        if isinstance(final_answer_prediction, str) == False:

            try:
                final_answer_truth = (
                    float(answer_truth) if "." in answer_truth else int(answer_truth)
                )

                # print(f"Predicted answer: {final_answer_prediction} \n")
                # print(f"True answer: {final_answer_truth} \n")

                if final_answer_prediction == final_answer_truth:
                    corrects += 1

                if save_response:
                    output_dict["index"].append(i)
                    output_dict["question"].append(question)
                    output_dict["answer"].append(final_answer_truth)
                    output_dict["prediction"].append(final_answer_prediction)

                readable_responses += 1
            except ValueError as e:
                print(f"Error: Value Error in iteration {i} \n")
                unsuccessful_responses_dict["index"].append(i)
                unsuccessful_responses_dict["input_prompt"].append(prompt)
                unsuccessful_responses_dict["response"].append(response)
                unsuccessful_responses_dict["error"].append(
                    "Cannot convert true answer"
                )

        else:
            print(f"Error: Value Error in iteration {i} \n")
            unsuccessful_responses_dict["index"].append(i)
            unsuccessful_responses_dict["input_prompt"].append(prompt)
            unsuccessful_responses_dict["response"].append(response)
            unsuccessful_responses_dict["error"].append(final_answer_prediction)

    if readable_responses == 0:
        accuracy = 0
    else:
        accuracy = corrects / readable_responses

    print(f"Accuracy: {accuracy}")

    if save_response:
        df_success = pd.DataFrame(output_dict)
        df_success.to_csv(f"{save_response}/successful_responses.csv", index=False)

        df_unsuccessful = pd.DataFrame(unsuccessful_responses_dict)
        df_unsuccessful.to_csv(
            f"{save_response}/unsuccessful_responses.csv", index=False
        )

    return accuracy, readable_responses


def evaluate_init(checkpoint):

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    checkpoint_path = checkpoint.replace("/", "_")
    output_dir = Path(rf"output/{checkpoint_path}_{now}")
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_kwargs = {
        "num_return_sequences": 1,
        "max_new_tokens": 256,
    }

    model, tokenizer = load_model(checkpoint)

    accuracy, readable_responses = evaluate(
        model,
        tokenizer,
        generate_kwargs,
        n_shots=8,
        subset="test",
        iterations=None,
        # iterations=1,
        save_response=output_dir,
    )

    config = model.config
    config = config.to_dict()

    results = {
        "model": checkpoint,
        "accuracy": accuracy,
        "num_responses": readable_responses,
        "generate_kwargs": generate_kwargs,
        "config": config,
        "prompt_avg": 32,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":

    config = init()
    print(config.device)
    logging.basicConfig(
        filename="./logs/running.log",
        filemode="a",
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    checkpoints = [
        "allenai/OLMo-1.7-7B-hf",
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ]

    for checkpoint in checkpoints:
        try:
            print(f"Running {checkpoint} \n")
            evaluate_init(checkpoint)
        except Exception as e:
            logging.error(f"Error: in {checkpoint} \n {e}")
        finally:
            torch.cuda.empty_cache()
