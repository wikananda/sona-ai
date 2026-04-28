from inspect import signature

from transformers import Seq2SeqTrainingArguments, TrainingArguments


def filter_training_args(arg_dict, task_type: str = "seq2seq"):
    """
    Filter a dict of training arguments to only the keys accepted by the
    relevant Hugging Face TrainingArguments class.
    """
    args_class = Seq2SeqTrainingArguments if task_type == "seq2seq" else TrainingArguments
    valid_keys = set(signature(args_class).parameters.keys())

    filtered_args = {}
    unused = []

    for key, value in arg_dict.items():
        if key in valid_keys:
            filtered_args[key] = value
        else:
            unused.append(key)

    if unused:
        print(f"Warning: Unused arguments: {unused}")

    return filtered_args

