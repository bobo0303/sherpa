from pydantic import BaseModel

#############################################################################


class ModlePath(BaseModel):
    # EN
    EN_encoder_path: str = (
        "models/gigaspeech/encoder-epoch-12-avg-2-chunk-16-left-64.onnx"
    )
    EN_decoder_path: str = (
        "models/gigaspeech/decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
    )
    EN_joiner_path: str = (
        "models/gigaspeech/joiner-epoch-12-avg-2-chunk-16-left-64.onnx"
    )
    EN_tokens_path: str = (
        "models/gigaspeech/tokens.txt"
    )
    EN_keywords_path: str = (
        "lib/en_hotwords.txt"
    )
    # ZH
    ZH_encoder_path: str = (
        "models/wenetspeech/encoder-epoch-12-avg-2-chunk-16-left-64.onnx"
    )
    ZH_decoder_path: str = (
        "models/wenetspeech/decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
    )
    ZH_joiner_path: str = (
        "models/wenetspeech/joiner-epoch-12-avg-2-chunk-16-left-64.onnx"
    )
    ZH_tokens_path: str = (
        "models/wenetspeech/tokens.txt"
    )
    ZH_keywords_path: str = (
        "lib/zh_hotwords.txt"
    )

    def get_paths(self, language: str):
        if language.upper() == "EN":
            return {
                "encoder_path": self.EN_encoder_path,
                "decoder_path": self.EN_decoder_path,
                "joiner_path": self.EN_joiner_path,
                "tokens_path": self.EN_tokens_path,
                "keywords_path": self.EN_keywords_path,
            }
        elif language.upper() == "ZH":
            return {
                "encoder_path": self.ZH_encoder_path,
                "decoder_path": self.ZH_decoder_path,
                "joiner_path": self.ZH_joiner_path,
                "tokens_path": self.ZH_tokens_path,
                "keywords_path": self.ZH_keywords_path,
            }
        else:
            raise ValueError("Unsupported language. Choose 'EN' or 'ZH'.")


#############################################################################

""" constant parameters """
class Config:
    def __init__(self):
        self.NUM_THREADS = 1
        self.MAX_ACTIVE_PATHS = 4
        self.KEYWORDS_SCORE = 1.0
        self.KEYWORDS_THRESHOLD = 0.01
        self.NUM_TRAILING_BLANKS = 1
        self.PROVIDER = "cpu"


#############################################################################
