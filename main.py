#@markdown # **Apple Podcast selection** 🎙️

#@markdown Enter the URL of the Apple Podcast you want to transcribe.

#@markdown ---
#@markdown #### **Apple Podcast**
URL = "https://podcasts.apple.com/cn/podcast/%E6%97%A0%E4%BA%BA%E7%9F%A5%E6%99%93/id1581271335?i=1000602067215" #@param {type:"string"}
#@markdown ---
#@markdown **Run this cell again if you change the video.**

import os
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse


def find_audio_url(html: str) -> str:
    # Find all .mp3 and .m4a URLs in the HTML content
    audio_urls = re.findall(r'https://[^\s^"]+(?:\.mp3|\.m4a)', html)

    # If there's at least one URL, return the first one
    if audio_urls:
        pattern = r'=https?://[^\s^"]+(?:\.mp3|\.m4a)'
        result = re.findall(pattern, audio_urls[-1])
        if result:
          return result[-1][1:]
        else:
          return audio_urls[-1]

    # Otherwise, return None
    return None

def get_file_extension(url: str) -> str:
    # Parse the URL to get the path
    parsed_url = urlparse(url)
    path = parsed_url.path

    # Extract the file extension using os.path.splitext
    _, file_extension = os.path.splitext(path)

    print("url", url, path, file_extension)
    # Return the file extension
    return file_extension

def download_apple_podcast(url: str, output_folder: str = 'downloads'):
    response = requests.get(url)
    if response.status_code != 200:
        print(
            f"Error: Unable to fetch the podcast page. Status code: {response.status_code}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    audio_url = find_audio_url(response.text)

    if not audio_url:
        print("Error: Unable to find the podcast audio url.")
        return

    episode_title = soup.find('span', {'class': 'product-header__title'})

    if not episode_title:
        print("Error: Unable to find the podcast title.")
        return

    episode_title = episode_title.text.strip().replace('/', '-')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = os.path.join(output_folder, f"{episode_title}{get_file_extension(audio_url)}")

    with requests.get(audio_url, stream=True) as r:
        r.raise_for_status()
        with open(output_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


    return episode_title, output_file


result = download_apple_podcast(URL)
if not result:
  print("Error: Unable to download podcast.")
else:
  (title, filepath) = result
  print(f"Downloaded podcast episode '{title}' to '{filepath}'")

#----------------------------------------------------------------------------------------------------------------------------------------------
#@markdown # **Model selection** 🧠

#@markdown As of the first public release, there are 4 pre-trained options to play with:

#@markdown |  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
#@markdown |:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
#@markdown |  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~32x      |
#@markdown |  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~16x      |
#@markdown | small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~6x       |
#@markdown | medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
#@markdown | large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |

#@markdown ---

Model = 'large-v2' #@param ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large', 'large-v2']
#@markdown ---
#@markdown **Run this cell again if you change the model.**

import whisper

whisper_model = whisper.load_model(Model)

if Model in whisper.available_models():
    print("**{Model} model is selected.")
else:
    print("{Model} model is no longer available. Please select one of the following: - {' - '.join(whisper.available_models())}")

#----------------------------------------------------------------------------------------------------------------------------------------------
#@markdown # **Run the model** 🚀

#@markdown Run this cell to execute the transcription of the video. This can take a while and very based on the length of the video and the number of parameters of the model selected above.

#@markdown ## **Parameters** ⚙️

#@markdown ### **Behavior control**
#@markdown ---
language = "Chinese"  # @param ['Auto detection', 'Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Assamese', 'Azerbaijani', 'Bashkir', 'Basque', 'Belarusian', 'Bengali', 'Bosnian', 'Breton', 'Bulgarian', 'Burmese', 'Castilian', 'Catalan', 'Chinese', 'Croatian', 'Czech', 'Danish', 'Dutch', 'English', 'Estonian', 'Faroese', 'Finnish', 'Flemish', 'French', 'Galician', 'Georgian', 'German', 'Greek', 'Gujarati', 'Haitian', 'Haitian Creole', 'Hausa', 'Hawaiian', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic', 'Indonesian', 'Italian', 'Japanese', 'Javanese', 'Kannada', 'Kazakh', 'Khmer', 'Korean', 'Lao', 'Latin', 'Latvian', 'Letzeburgesch', 'Lingala', 'Lithuanian', 'Luxembourgish', 'Macedonian', 'Malagasy', 'Malay', 'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Moldavian', 'Moldovan', 'Mongolian', 'Myanmar', 'Nepali', 'Norwegian', 'Nynorsk', 'Occitan', 'Panjabi', 'Pashto', 'Persian', 'Polish', 'Portuguese', 'Punjabi', 'Pushto', 'Romanian', 'Russian', 'Sanskrit', 'Serbian', 'Shona', 'Sindhi', 'Sinhala', 'Sinhalese', 'Slovak', 'Slovenian', 'Somali', 'Spanish', 'Sundanese', 'Swahili', 'Swedish', 'Tagalog', 'Tajik', 'Tamil', 'Tatar', 'Telugu', 'Thai', 'Tibetan', 'Turkish', 'Turkmen', 'Ukrainian', 'Urdu', 'Uzbek', 'Valencian', 'Vietnamese', 'Welsh', 'Yiddish', 'Yoruba']
#@markdown > Language spoken in the audio, use `Auto detection` to let Whisper detect the language.
#@markdown ---
verbose = 'Live transcription' #@param ['Live transcription', 'Progress bar', 'None']
#@markdown > Whether to print out the progress and debug messages.
#@markdown ---
output_format = 'srt' #@param ['txt', 'vtt', 'srt', 'tsv', 'json', 'all']
#@markdown > Type of file to generate to record the transcription.
#@markdown ---
task = 'transcribe' #@param ['transcribe', 'translate']
#@markdown > Whether to perform X->X speech recognition (`transcribe`) or X->English translation (`translate`).
#@markdown ---

#@markdown 

#@markdown ### **Optional: Fine tunning** 
#@markdown ---
temperature = 0.15 #@param {type:"slider", min:0, max:1, step:0.05}
#@markdown > Temperature to use for sampling.
#@markdown ---
temperature_increment_on_fallback = 0.2 #@param {type:"slider", min:0, max:1, step:0.05}
#@markdown > Temperature to increase when falling back when the decoding fails to meet either of the thresholds below.
#@markdown ---
best_of = 5 #@param {type:"integer"}
#@markdown > Number of candidates when sampling with non-zero temperature.
#@markdown ---
beam_size = 8 #@param {type:"integer"}
#@markdown > Number of beams in beam search, only applicable when temperature is zero.
#@markdown ---
patience = 1.0 #@param {type:"number"}
#@markdown > Optional patience value to use in beam decoding, as in [*Beam Decoding with Controlled Patience*](https://arxiv.org/abs/2204.05424), the default (1.0) is equivalent to conventional beam search.
#@markdown ---
length_penalty = -0.05 #@param {type:"slider", min:-0.05, max:1, step:0.05}
#@markdown > Optional token length penalty coefficient (alpha) as in [*Google's Neural Machine Translation System*](https://arxiv.org/abs/1609.08144), set to negative value to uses simple length normalization.
#@markdown ---
suppress_tokens = "-1" #@param {type:"string"}
#@markdown > Comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations.
#@markdown ---
initial_prompt = "" #@param {type:"string"}
#@markdown > Optional text to provide as a prompt for the first window.
#@markdown ---
condition_on_previous_text = True #@param {type:"boolean"}
#@markdown > if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop.
#@markdown ---
fp16 = True #@param {type:"boolean"}
#@markdown > whether to perform inference in fp16.
#@markdown ---
compression_ratio_threshold = 2.4 #@param {type:"number"}
#@markdown > If the gzip compression ratio is higher than this value, treat the decoding as failed.
#@markdown ---
logprob_threshold = -1.0 #@param {type:"number"}
#@markdown > If the average log probability is lower than this value, treat the decoding as failed.
#@markdown ---
no_speech_threshold = 0.6 #@param {type:"slider", min:-0.0, max:1, step:0.05}
#@markdown > If the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence.
#@markdown ---

verbose_lut = {
    'Live transcription': True,
    'Progress bar': False,
    'None': None
}

import numpy as np
import warnings
import shutil
from pathlib import Path

args = dict(
    language = (None if language == "Auto detection" else language),
    verbose = verbose_lut[verbose],
    task = task,
    temperature = temperature,
    temperature_increment_on_fallback = temperature_increment_on_fallback,
    best_of = best_of,
    beam_size = beam_size,
    patience=patience,
    length_penalty=(length_penalty if length_penalty>=0.0 else None),
    suppress_tokens=suppress_tokens,
    initial_prompt=(None if not initial_prompt else initial_prompt),
    condition_on_previous_text=condition_on_previous_text,
    fp16=fp16,
    compression_ratio_threshold=compression_ratio_threshold,
    logprob_threshold=logprob_threshold,
    no_speech_threshold=no_speech_threshold
)

temperature = args.pop("temperature")
temperature_increment_on_fallback = args.pop("temperature_increment_on_fallback")
if temperature_increment_on_fallback is not None:
    temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
else:
    temperature = [temperature]

if Model.endswith(".en") and args["language"] not in {"en", "English"}:
    warnings.warn(f"{Model} is an English-only model but receipted '{args['language']}'; using English instead.")
    args["language"] = "en"

print("{filepath}")

audio_path_local = Path(filepath).resolve()
print("audio local path:", audio_path_local)

transcription = whisper.transcribe(
    whisper_model,
    str(audio_path_local),
    temperature=temperature,
    **args,
)

# Save output
whisper.utils.get_writer(
    output_format=output_format,
    output_dir=audio_path_local.parent
)(
    transcription,
    title
)

try:
    if output_format=="all":
        for ext in ('txt', 'vtt', 'srt', 'tsv', 'json'):
            transcript_file_name = title + "." + ext
            print("Transcript file created: {transcript_file_name}")
    else:
        transcript_file_name = title + "." + output_format

        print("Transcript file created: {transcript_file_name}")

except:
    print("Transcript file created: {transcript_local_path}")
