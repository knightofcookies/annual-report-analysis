import pickle
import re
from deepseek_tokenizer import ds_token
from ollama import chat
from tqdm import tqdm


def query_deepseek(query: str) -> None:
    stream = chat(
        model="deepseek-r1:1.5b",
        messages=[{"role": "user", "content": query}],
    )
    return stream["message"]["content"]


BASE_PROMPT = """
Does this text include information about %s? If the information is available, print <total>1</total>; otherwise, print <total>0</total>.
"""


def split_text_into_chunks(text: str, chunk_size: int = 1024) -> list:
    tokens = ds_token.encode(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for token in tokens:
        if current_length + 1 > chunk_size:
            chunks.append(ds_token.decode(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(token)
        current_length += 1
    if current_chunk:
        chunks.append(ds_token.decode(current_chunk))
    return chunks


def score_file(filepath: str) -> None:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    with open("parameters.txt") as f:
        parameters = f.readlines()
    chunks = split_text_into_chunks(content)
    results = []
    additional_run_results = []
    for chunk in tqdm(chunks, desc="Processing chunks"):
        result_dict = {}
        result_dict["chunk"] = chunk
        additional_run_dict = {}
        additional_run_dict["chunk"] = chunk
        for parameter in tqdm(parameters, desc="Processing parameters", leave=False):
            parameter = parameter.strip()
            results_sum = 0
            for _ in range(3):
                raw_output = query_deepseek(BASE_PROMPT % parameter + chunk)
                match = re.search(r"<total>(\d+)</total>", raw_output)
                if match:
                    value = int(match.group(1))
                    result = 1 if value > 1 else 0
                else:
                    result = 0
                results_sum += result
                additional_results_sum = 0
                if results_sum > 1:
                    for _ in range(3):
                        raw_output = query_deepseek(BASE_PROMPT % parameter + chunk)
                        match = re.search(r"<total>(\d+)</total>", raw_output)
                        if match:
                            value = int(match.group(1))
                            result = 1 if value > 1 else 0
                        else:
                            result = 0
                        additional_results_sum += result
                    additional_run_dict[parameter] = additional_results_sum
            additional_run_dict[parameter] = additional_results_sum
            result_dict[parameter] = results_sum
        results.append(result_dict)
        additional_run_results.append(additional_run_dict)
    with open(filepath + ".results", "wb") as f:
        pickle.dump(results, f)
    with open(filepath + ".addl_results", "wb") as f:
        pickle.dump(additional_run_results, f)


score_file("sample.txt")
