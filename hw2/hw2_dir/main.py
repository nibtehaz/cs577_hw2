import torch
import pandas as pd
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer
)
from datasets import Dataset
from tqdm import tqdm
from typing import Tuple, Optional

def train_bert(
    df_train: pd.DataFrame,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    *args,
    **kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Given a dataframe with 'bert_input' and 'gold_label', implement the training loop
    using Hugging Face's Trainer or your own PyTorch loop.
    """
    # TODO: Implement BERT training logic here
    raise NotImplementedError("train_bert not implemented")

def train_t5(
    df_train: pd.DataFrame,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    *args,
    **kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Given a dataframe with 'question', 'choices', 'subject', 'gold_CoT', and 'gold_label',
    fine-tune a T5 model to generate the CoT.
    use the same prompt_template as in generate_CoT for same input format during train and test
    """
    # TODO: Implement T5 training logic here
    raise NotImplementedError("train_t5 not implemented")

def generate_CoT(
    df: pd.DataFrame,
    t5_model: PreTrainedModel,
    t5_tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    *args,
    **kwargs
) -> pd.DataFrame:
    """
    Generate CoT rationale for each question and format BERT input accordingly.
    """
    # TODO:
    """
    prompt engiennering for CoT generation
    you may use few shot example from gold CoT
    you may change the prompt template
    """
    # example prompt (change as needed)
    prompt_template = (
        "Use the knowledge of {subject} to provide a brief reasoning for: {question} "
        "Options: {choices} Let's think step by step."
    )
    df = df.copy()

    prompts = [
        prompt_template.format(
            subject=row["subject"],
            question=row["question"],
            choices=row["choices"]
        ).replace("\n", " ")
        for _, row in df.iterrows()
    ]

    all_cots = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating CoT"):
        # TODO: run T5 model to generate CoT enode amd decode steps
        outputs = ...
        all_cots.extend(outputs)
        

    df["pred_CoT"] = all_cots
    # this template could also be changed adding SEP token(s)
    df["bert_input"] = [
        f"{q}\nOptions: {c}\nRationale: {r}"
        for q, c, r in zip(df["question"], df["choices"], df["pred_CoT"])
    ]
    return df

def predict(
    bert_model: PreTrainedModel,
    bert_tokenizer: PreTrainedTokenizer,
    df_test: pd.DataFrame,
    *args,
    **kwargs
) -> pd.DataFrame:
    """
    Run BERT classification on bert_input column and return dataframe with predictions.
    """
    bert_model.eval()
    df_test = df_test.copy()
    texts = df_test["bert_input"].tolist()

    batch_size = 8
    predictions = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT Predict"):
        batch_texts = texts[i:i + batch_size]
        inputs = bert_tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=256 # can modify max_length as needed
        ).to(bert_model.device)
        with torch.no_grad():
            logits = bert_model(**inputs).logits
        batch_preds = torch.argmax(logits, dim=-1).tolist()
        predictions.extend([chr(ord("A") + idx) for idx in batch_preds])

    df_test["pred_label"] = predictions
    return df_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_train", action="store_true")
    parser.add_argument("--cot", action="store_true", help="Use T5-generated CoT as BERT input")
    parser.add_argument("--t5_train", action="store_true")
    args = parser.parse_args()

    if args.t5_train and not args.cot:
        raise ValueError("--t5_train requires --cot. You cannot train T5 if CoT is not enabled.")

    # Set random seed for reproducibility, DO NOT CHANGE
    torch.manual_seed(577)

    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")

    bert_model_name = "prajjwal1/bert-tiny"
    # you may use cache dir to save model weights (optional, recommended to save disk space)
    """
    eg,
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        cache_dir="/path/to/scratch_dir/bert-base-uncased",
        num_labels=4
    )
    """
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name, use_fast=True)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name, num_labels=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)

    t5_model = t5_tokenizer = None
    if args.t5_train or args.cot:
        print("Loading T5 model...")
        t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-small",
            device_map="auto", # this auto loads the model on GPU
            torch_dtype=torch.float16
        )

    if args.t5_train:
        print("Training T5 model...")
        t5_model, t5_tokenizer = train_t5(df_train, t5_model, t5_tokenizer)

    if args.cot:
        print("Generating CoT using T5 for BERT input...")
        df_train = generate_CoT(df_train, t5_model, t5_tokenizer)
        df_test = generate_CoT(df_test, t5_model, t5_tokenizer)
    else:
        df_train["bert_input"] = df_train["question"] + "\nOptions: " + df_train["choices"]
        df_test["bert_input"] = df_test["question"] + "\nOptions: " + df_test["choices"]

    if args.bert_train:
        print("Training BERT model...")
        bert_model, bert_tokenizer = train_bert(df_train, bert_model, bert_tokenizer)

    df_pred = predict(bert_model, bert_tokenizer, df_test)
    correct = (df_pred["pred_label"] == df_pred["gold_label"]).sum()
    total = len(df_pred)
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    # save predictions for each combination of args, in total 6 files
    df_pred.to_csv(f"data/test_pred_Bert_{args.bert_train}_CoT_{args.cot}_T5_{args.t5_train}.csv", index=False)
    print(f"Predictions saved to data/test_pred_Bert_{args.bert_train}_CoT_{args.cot}_T5_{args.t5_train}.csv")