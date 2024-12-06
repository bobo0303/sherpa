#!/usr/bin/env python3

"""
This script encodes the texts to tokens and writes the results to the output file.
Configure the parameters directly within the script.
"""

from sherpa_onnx import text2token
import sys 
import argparse  

# 創建解析器  
parser = argparse.ArgumentParser(description="Set language parameter (EN or ZH)")  
parser.add_argument("language", choices=["EN", "ZH"], help="Language code: 'EN' for English, 'ZH' for Chinese")  
  
# 解析參數  
args = parser.parse_args()  
language = args.language.lower() 

language = sys.argv[1]  


if language.lower() == "en":
    # Set parameters here
    TEXT_PATH = "lib/en_texts.txt"  # Path to the input texts
    TOKENS_PATH = "models/gigaspeech/tokens.txt"  # Path to tokens.txt
    TOKENS_TYPE = "bpe"  # Type of modeling units (choose from: cjkchar, bpe, cjkchar+bpe, fpinyin, ppinyin)
    BPE_MODEL_PATH = "models/gigaspeech/bpe.model"  # Path to bpe.model (required for bpe or cjkchar+bpe)
    OUTPUT_PATH = "lib/en_hotwords.txt"  # Path where the encoded tokens will be written to
elif language.lower() == "zh":
    # Set parameters here
    TEXT_PATH = "lib/zh_texts.txt"  # Path to the input texts
    TOKENS_PATH = "models/wenetspeech/tokens.txt"  # Path to tokens.txt
    TOKENS_TYPE = "ppinyin"  # Type of modeling units (choose from: cjkchar, bpe, cjkchar+bpe, fpinyin, ppinyin)
    OUTPUT_PATH = "lib/zh_hotwords.txt"  # Path where the encoded tokens will be written to
else:
    print("="*10 + " language error (please choose \"EN\" or \"ZH\") " + "="*10)
    sys.exit(1)  # Exit the program with a non-zero status to indicate an error  
    
print(OUTPUT_PATH)


def main():
    texts = []
    extra_info = []

    # Read the input texts file and separate text and extra information
    with open(TEXT_PATH, "r", encoding="utf8") as f:
        for line in f:
            extra = []
            text = []
            toks = line.strip().split()
            for tok in toks:
                if tok[0] in {":", "#", "@"}:
                    extra.append(tok)
                else:
                    text.append(tok)
            texts.append(" ".join(text))
            extra_info.append(extra)

    # Encode texts using the sherpa_onnx text2token function
    encoded_texts = text2token(
        texts,
        tokens=TOKENS_PATH,
        tokens_type=TOKENS_TYPE,
        bpe_model=BPE_MODEL_PATH if TOKENS_TYPE in {"bpe", "cjkchar+bpe"} else None,
    )

    # Write encoded tokens with extra info to the output file
    with open(OUTPUT_PATH, "w", encoding="utf8") as f:
        for i, txt in enumerate(encoded_texts):
            txt += extra_info[i]
            f.write(" ".join(txt) + "\n")


if __name__ == "__main__":
    main()
