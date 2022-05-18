from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
import torch
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


batch_size = 64
df_docs = pd.read_csv("data/lab_docs.csv")
examples = df_docs["text"].to_list()


def generate_summaries(max_src_length: str, max_tgt_len: str):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(
        device
    )
    fout = open("data/lab_titles", "w", encoding="utf-8")
    for batch in tqdm(list(chunks(examples, batch_size))):
        dct = tokenizer.batch_encode_plus(
            batch,
            max_length=max_src_length,
            return_tensors="pt",
            pad_to_max_length=True,
        ).to(device)
        summaries = model.generate(
            num_beams=1, max_length=max_tgt_len, early_stopping=True, **dct
        )
        dec = tokenizer.batch_decode(
            summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for hypothesis in dec:
            hypothesis = hypothesis.replace('"', "")
            fout.write(hypothesis.strip() + "\n")
            fout.flush()

if __name__ == '__main__':
    print(generate_summaries(512, 10))