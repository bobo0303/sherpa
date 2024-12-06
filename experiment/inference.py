import sherpa_onnx
from pathlib import Path
import numpy as np
import time
import wave
from typing import List, Tuple


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html to download it"
    )


tokens_path = "models/wenetspeech/tokens.txt"
encoder_path = "models/gigaspeech/encoder-epoch-12-avg-2-chunk-16-left-64.onnx"
decoder_path = "models/gigaspeech/decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
joiner_path = "models/gigaspeech/joiner-epoch-12-avg-2-chunk-16-left-64.onnx"
num_threads = 1
max_active_paths = 4
# keywords_path = 'models/gigaspeech/keywords.txt'
# keywords_path = "models/gigaspeech/test_wavs/test_keywords.txt"
keywords_path = "hotwords.txt"
keywords_score = 1.0
keywords_threshold = 0.1
num_trailing_blanks = 1
provider = "cpu"

keyword_spotter = sherpa_onnx.KeywordSpotter(
    tokens=tokens_path,
    encoder=encoder_path,
    decoder=decoder_path,
    joiner=joiner_path,
    num_threads=num_threads,
    max_active_paths=max_active_paths,
    keywords_file=keywords_path,
    keywords_score=keywords_score,
    keywords_threshold=keywords_threshold,
    num_trailing_blanks=num_trailing_blanks,
    provider=provider,
)


def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    Args:
      wave_filename:
        Path to a wave file. It should be single channel and each sample should
        be 16-bit. Its sample rate does not need to be 16kHz.
    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples, which are
       normalized to the range [-1, 1].
       - sample rate of the wave file
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


sound_files = [
    "audio/hey_check/hey_check_35.wav",
]


print("Started!")
start_time = time.time()

streams = []
total_duration = 0
for wave_filename in sound_files:

    assert_file_exists(wave_filename)
    samples, sample_rate = read_wave(wave_filename)
    duration = len(samples) / sample_rate
    total_duration += duration

    s = keyword_spotter.create_stream()

    s.accept_waveform(sample_rate, samples)

    tail_paddings = np.zeros(int(0.66 * sample_rate), dtype=np.float32)
    s.accept_waveform(sample_rate, tail_paddings)

    s.input_finished()

    streams.append(s)

hey = 0
check = 0
results = [""] * len(streams)
while True:
    ready_list = []
    for i, s in enumerate(streams):
        if keyword_spotter.is_ready(s):
            ready_list.append(s)
        r = keyword_spotter.get_result(s)
        if r:
            results[i] += f"{r}/"
            print(f"{r} is detected.")
            if r == "HEY":
                hey += 1
            if r == "CHECK":
                check += 1
    if len(ready_list) == 0:
        break
    keyword_spotter.decode_streams(ready_list)
end_time = time.time()
print("Done!")

for wave_filename, result in zip(sound_files, results):
    print(f"{wave_filename}\n{result}")
    print("-" * 10)

elapsed_seconds = end_time - start_time
rtf = elapsed_seconds / total_duration
print(f"num_threads: {num_threads}")
print(f"Wave duration: {total_duration:.3f} s")
print(f"Elapsed time: {elapsed_seconds:.3f} s")
print(f"Real time factor (RTF): {elapsed_seconds:.3f}/{total_duration:.3f} = {rtf:.3f}")
print(f"HEY num: {hey} | CHECK num: {check}")
