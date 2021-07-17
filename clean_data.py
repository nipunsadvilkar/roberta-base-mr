#!/usr/bin/env python
# coding=utf-8
"""
Script to clean mc4 marathi dataset and 
writing new text file with only marathi documents
"""
import string
import argparse

# import re
import regex as re  # pip install regex

from datasets import load_dataset

# following regex requires third party regex module
devanagari_pattern = re.compile(r'[\p{Devanagari}\s!"\#%\'\(\)\-\?\[\]\.,\|]+')
punct_rm = re.compile("[" + re.escape(string.punctuation) + "]")


def clean_marathi_text_unicode_approach(text):
    """
    works with python inbuilt re module
    """
    clean_text = "".join([tok.group().strip() for tok in re.finditer(r'[\u0900-\u097F\s]', text)])
    clean_text = re.sub(r"\s+", " ", clean_text)
    return clean_text

def clean_marathi_text_regex(text):
    """
    requires third party regex module
    """
    # capture only marathi words, letters, numbers characters
    cleaned = "".join([tok.group().strip() for tok in re.finditer(devanagari_pattern, text)])
    # remove consecutive space with single space
    cleaned = re.sub(r" +", " ", cleaned)
    # remove all puncts to see
    only_contains_mr_chars = re.sub(punct_rm, "", cleaned)
    return cleaned, only_contains_mr_chars.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create clean marathi dataset in text format.')
    parser.add_argument('--split_type', type=str, default='validation', help="Split type of dataset. Should be `train` or `validation`")
    parser.add_argument('--keep_punct', type=bool, default=False, help="Keep of remove puctuations from dataset")

    args = parser.parse_args()
    split_type = args.split_type
    keep_punct = args.keep_punct

    mr_mc4 = load_dataset("mc4", "mr", split=split_type)

    print(mr_mc4)

    marathi_examples_count = 0
    output_filename = f"mr_{split_type}.txt" if keep_punct else f"mr_{split_type}_rmpunct.txt"
    with open(output_filename, 'w', encoding="utf-8") as f:
        for ind, doc in enumerate(mr_mc4):
            clean_text_with_punct, only_contains_mr_chars = clean_marathi_text_regex(doc["text"])
            if only_contains_mr_chars:
                marathi_examples_count += 1
                doc = clean_text_with_punct.strip() if keep_punct else only_contains_mr_chars.strip()
                f.write(doc)
                f.write("\n")
    print(f"{split_type} clean docs count {marathi_examples_count} out of {len(mr_mc4)}")
    # train clean docs count 1581396 
    # valid clean docs count 1578