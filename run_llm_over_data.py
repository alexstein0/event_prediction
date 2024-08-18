import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datasets
import logging
import json
import os
import torch.nn.functional as F

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
log = logging.getLogger(__name__)

torch.manual_seed(0)


def remove_label(prev_seq: str, question: str):
    return " ".join([x for x in prev_seq.split() if not x.startswith(question)])


def extract_answer(seq: str):
    splitted = seq.split()
    prev_seq = " ".join(splitted[0:-2])
    question, answer = splitted[-2].split(":")
    return prev_seq, question, answer


def create_prompts(question, prev_seq):
    system_prompt = "You are a helpful AI assistant that is good at analyzing sequential tabular data.  \
The user will give you tabular data in the form '<column name>:<value>' with '[ROW]' indicating the end of a row.  \
You will be given the next column name and are tasked with predicting the corresponding value. \
Please only respond with the next value. "

    user_prompt = f"{prev_seq}"
    # user_prompt = f"{prev_seq} {question}:"
    assistent_response = f"{question}:"
    return system_prompt, user_prompt, assistent_response


def create_message(system_prompt, user_prompt, assistent_response):
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
        {'role': 'assistant', 'content': assistent_response},
    ]
    return messages


def create_input_payload(message_dict, tok, max_length):
    formatted_text = tok.apply_chat_template(message_dict, tokenize=False, add_generation_prompt=True)
    # print(formatted_text)
    inputs = tok(formatted_text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length,
                 add_special_tokens=False)
    # inputs = tok(formatted_text, return_tensors='pt', padding=True, truncation=True, max_length=512, add_special_tokens=False)
    return inputs


def get_prediction_probs(mod, input_ids, attention_mask, tokenizer, target_tokens):
    # only works for amazon right now, needs to change based on targets
    hidden = mod(input_ids, attention_mask=attention_mask)
    logits = hidden.logits[:, -1, target_tokens]
    probabilities = F.softmax(logits, dim=-1).cpu().tolist()
    # decoded = tok.decode(outputs[0][input_dict['input_ids'].size(1):], skip_special_tokens=True)
    outputs = [{"probs": x} for x in probabilities]
    return outputs


def generate_predicted_text(mod, input_ids, attention_mask, tokenizer):
    gen_ids = mod.generate(input_ids, attention_mask=attention_mask, max_new_tokens=64,
                           pad_token_id=tokenizer.eos_token_id)
    gen_ids = gen_ids.cpu()
    answer_start = input_ids.size(1)
    outputs = []
    for out_ind in range(len(gen_ids)):
        out = gen_ids[out_ind]
        prediction = tokenizer.decode(out[answer_start:], skip_special_tokens=True)
        prefix = 'assistant'
        if prediction.startswith(prefix):
            # this is hack because of padding artificially adding these tokens when generate is called
            prediction = prediction[len(prefix):].lstrip()
        outputs.append({"pred": prediction})
    return outputs


def save_outputs(data, save_dir, save_name):
    # Save the list of dictionaries as a JSON file
    save_name = f"{save_name}.json"
    file_name = os.path.join(save_dir, save_name)
    # Check if the directory exists
    if not os.path.exists(save_dir):
        # Create the directory
        os.makedirs(save_dir)

    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)  # `indent=4` for pretty-printing


def main(args):

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name

    dataset_path = os.path.join(dataset_dir, dataset_name)

    ds = datasets.load_from_disk(dataset_path)

    # Set the model to evaluation mode
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token

    # max_len = 512
    max_len = 1024
    model.eval()

    # Define the batch size
    batch_size = args.batch_size

    # Run inference in batches
    results = []
    max_samples = len(ds)
    # max_samples = 8
    log.info(f"{max_samples} total samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # target_tokens = [2575, 4139]
    # [True, False]
    # target_tokens = [16, 17, 18, 19, 20]
    # [0, 1]
    target_tokens = [15, 16]

    for i in range(0, max_samples - batch_size, batch_size):
        if i % (batch_size * 100) == 0:
            log.info(f"{i} / {max_samples}")
        batch = ds.select(range(i, i + batch_size))
        batch_inputs = []
        for sample in batch:
            sample = sample["text"]
            prev_seq, question, answer = extract_answer(sample)
            if args.exclude_labels:
                prev_seq = remove_label(prev_seq, question)
            if args.replace_question is not None:
                question = args.replace_question
            system_prompt, user_prompt, assistent_response = create_prompts(question, prev_seq)
            messages = create_message(system_prompt, user_prompt, assistent_response)
            inputs = create_input_payload(messages, tokenizer, max_length=max_len)
            inputs["answer"] = answer
            batch_inputs.append(inputs)

        input_ids = torch.cat([inputs['input_ids'] for inputs in batch_inputs], dim=0)
        attention_mask = torch.cat([inputs['attention_mask'] for inputs in batch_inputs], dim=0)
        answers = [inputs['answer'] for inputs in batch_inputs]
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = get_prediction_probs(model, input_ids, attention_mask, tokenizer, target_tokens)
            # outputs = generate_predicted_text(model, input_ids, attention_mask, tokenizer)

        for o in range(len(outputs)):
            metrics_dict = outputs[o]
            metrics_dict["label"] = answers[o]
        results.extend(outputs)

    log.info(f"SAVING RESULTS: {len(results)}")
    output_dir = os.path.join("outputs", args.output_dir)
    save_outputs(results, output_dir, args.dataset_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='text_results2')
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--exclude_labels', type=bool, default=False)
    parser.add_argument('--replace_question', type=str)
    args_dict = parser.parse_args()
    if args_dict.replace_question is not None:
        args_dict.replace_question = " ".join(args_dict.replace_question.split("|"))
    print(args_dict)
    main(args_dict)
