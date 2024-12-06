# model.py
import os
import gc
import sys
import time
import torch
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.constant import ModlePath, Config

import sherpa_onnx
from pathlib import Path
import numpy as np
import wave
from typing import List, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Model:
    def __init__(self):
        """
        Initialize the Model class with default attributes.
        """
        self.model = None
        self.keyword_spotter = None
        self.models_path = ModlePath()
        self.config = Config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_parameter = {
            "model": None,
            "disable_update": True,
            "disable_pbar": True,
            "device": self.device,
        }

    def load_model(self, language):
        """  
        Load the specified model based on the model's name.  
  
        Parameters:  
        ----------  
        language: str  
            The language of the model to be loaded.  
  
        Returns:  
        ----------  
        None  
  
        Logs:  
        ----------  
        Loading status and time.  
        """  

        # 實現模型載入的邏輯
        start = time.time()

        model_config = self.models_path.get_paths(language)
        self._release_model()
        
        self.keyword_spotter = sherpa_onnx.KeywordSpotter(
            tokens=model_config['tokens_path'],
            encoder=model_config['encoder_path'],
            decoder=model_config['decoder_path'],
            joiner=model_config['joiner_path'],
            keywords_file=model_config['keywords_path'],
            num_threads=self.config.NUM_THREADS,
            max_active_paths=self.config.MAX_ACTIVE_PATHS,
            keywords_score=self.config.KEYWORDS_SCORE,
            keywords_threshold=self.config.KEYWORDS_THRESHOLD,
            num_trailing_blanks=self.config.NUM_TRAILING_BLANKS,
            provider=self.config.PROVIDER,
        )
        
        self.model = self.keyword_spotter.create_stream()
        

        end = time.time()

        logger.info(f"'{language}' Model has been loaded in {end - start:.2f} seconds.")

    def _release_model(self):
        """
        Release the resources occupied by the current model.

        :param
        ----------
        None: The function does not take any parameters.

        :rtype
        ----------
        None: The function does not return any value.

        :logs
        ----------
        Model release status.
        """
        if self.model is not None:
            del self.model
            del self.keyword_spotter
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Previous model resources have been released.")
            
    def read_wave(self, wave_filename: str) -> Tuple[np.ndarray, int]:
        """  
        Read a wave file.  
  
        Parameters:  
        ----------  
        wave_filename: str  
            Path to a wave file. It should be single channel and each sample should be 16-bit. Its sample rate does not need to be 16kHz.  
  
        Returns:  
        ----------  
        tuple:   
            - A 1-D array of dtype np.float32 containing the samples, which are normalized to the range [-1, 1].  
            - Sample rate of the wave file.  
        """  

        with wave.open(wave_filename) as f:
            assert f.getnchannels() == 1, f.getnchannels()
            assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
            num_samples = f.getnframes()
            samples = f.readframes(num_samples)
            samples_int16 = np.frombuffer(samples, dtype=np.int16)
            samples_float32 = samples_int16.astype(np.float32)

            samples_float32 = samples_float32 / 32768
            return samples_float32, f.getframerate()
        
    def assert_file_exists(self, filename: str):
        assert Path(filename).is_file(), (f"{filename} does not exist!")

    def transcribe(self, audio_file_path):
        """  
        Perform transcription on the given audio file.  
  
        Parameters:  
        ----------  
        audio_file_path: str  
            The path to the audio file to be transcribed.  
  
        Returns:  
        ----------  
        tuple:   
            A tuple containing a dictionary with hotwords and the inference time.  
  
        Logs:  
        ----------  
        Inference status and time.  
        """  

        # 實現推論的邏輯
        start = time.time()
        self.assert_file_exists(audio_file_path)
        samples, sample_rate = self.read_wave(audio_file_path)
        duration = len(samples) / sample_rate


        self.model.accept_waveform(sample_rate, samples)

        tail_paddings = np.zeros(int(0.66 * sample_rate), dtype=np.float32)
        self.model.accept_waveform(sample_rate, tail_paddings)

        self.model.input_finished()
        
        hotword = []
        while True:
            ready_list = []
            if self.keyword_spotter.is_ready(self.model):
                ready_list.append(self.model)
            r = self.keyword_spotter.get_result(self.model)
            if r:
                hotword.append(r)
                logger.debug(f"{r} is detected.")
            
            if len(ready_list) == 0:
                break   
            self.keyword_spotter.decode_streams(ready_list)
            
        end = time.time()
        inference_time = end - start
        
        logger.debug(f"inference time {inference_time} secomds.")

        return {
            "hotword": hotword,
        }, inference_time
