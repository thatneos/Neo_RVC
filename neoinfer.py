import argparse
import gradio as gr
import requests
import random
import os
import zipfile 
import librosa
import time
from infer_rvc_python import BaseLoader
from pydub import AudioSegment
from audio_separator.separator import Separator


separator = Separator()
converter = BaseLoader(only_cpu=False, hubert_path=None, rmvpe_path=None)

# CONFIGS
TEMP_DIR = "temp"
MODEL_PREFIX = "model"
PITCH_ALGO_OPT = [
    "pm",
    "harvest",
    "crepe",
    "rmvpe",
    "rmvpe+",
]


os.makedirs(TEMP_DIR, exist_ok=True)

def unzip_file(file):
    filename = os.path.basename(file).split(".")[0]
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(TEMP_DIR, filename))
    return True

def progress_bar(total, current):
    return "[" + "=" * int(current / total * 20) + ">" + " " * (20 - int(current / total * 20)) + "] " + str(int(current / total * 100)) + "%"

def contains_bad_word(text, bad_words):
    text_lower = text.lower()
    for word in bad_words:
        if word.lower() in text_lower:
            return True
    return False

def download_from_url(url, name=None):
    if name is None:
        raise ValueError("The model name must be provided")
    if "/blob/" in url:
        url = url.replace("/blob/", "/resolve/")
    if "huggingface" not in url:
        return ["The URL must be from huggingface", "Failed", "Failed"]
    filename = os.path.join(TEMP_DIR, MODEL_PREFIX + str(random.randint(1, 1000)) + ".zip")
    response = requests.get(url)
    total = int(response.headers.get('content-length', 0))
    if total > 500000000:
        return ["The file is too large. You can only download files up to 500 MB in size.", "Failed", "Failed"]
    current = 0
    with open(filename, "wb") as f:
        for data in response.iter_content(chunk_size=4096):
            f.write(data)
            current += len(data)
            print(progress_bar(total, current), end="\r")
    try:
        unzip_file(filename)
    except Exception as e:
        return ["Failed to unzip the file", "Failed", "Failed"]
    unzipped_dir = os.path.join(TEMP_DIR, os.path.basename(filename).split(".")[0])
    pth_files = []
    index_files = []
    for root, dirs, files in os.walk(unzipped_dir):
        for file in files:
            if file.endswith(".pth"):
                pth_files.append(os.path.join(root, file))
            elif file.endswith(".index"):
                index_files.append(os.path.join(root, file))
    if name == "":
        name = pth_file.split(".")[0]
    MODELS.append({"model": pth_file, "index": index_file, "model_name": name})
    return ["Downloaded as " + name, pth_files[0], index_files[0]]

def inference(audio, model_name):
    output_data = inf_handler(audio, model_name)
    vocals = output_data[0]
    inst = output_data[1]
    return vocals, inst

def convert_now(audio_files, random_tag, converter):
    return converter(
        audio_files,
        random_tag,
        overwrite=False,
        parallel_workers=8
    )

def inf_handler(audio, model_name):
    separator.load_model()
    output_files = separator.separate(audio)
    vocals = output_files[0]
    inst = output_files[1]
    return vocals, inst

def run(
    model,
    audio_files,
    pitch_alg,
    pitch_lvl,
    index_inf,
    r_m_f,
    e_r,
    c_b_p,
):
    if not audio_files:
        raise ValueError("The audio pls")
    if isinstance(audio_files, str):
        audio_files = [audio_files]
    try:
        duration_base = librosa.get_duration(filename=audio_files[0])
        print("Duration:", duration_base)
    except Exception as e:
        print(e)
    random_tag = "USER_"+str(random.randint(10000000, 99999999))
    file_m = model
    print("File model:", file_m)
    # get from MODELS
    for model in MODELS:
        if model["model_name"] == file_m:
            print(model)
            file_m = model["model"]
            file_index = model["index"]
            break
    if not file_m.endswith(".pth"):
        raise ValueError("The model file must be a .pth file")
    print("Random tag:", random_tag)
    print("File model:", file_m)
    print("Pitch algorithm:", pitch_alg)
    print("Pitch level:", pitch_lvl)
    print("File index:", file_index)
    print("Index influence:", index_inf)
    print("Respiration median filtering:", r_m_f)
    print("Envelope ratio:", e_r)
    converter.apply_conf(
        tag=random_tag,
        file_model=file_m,
        pitch_algo=pitch_alg,
        pitch_lvl=pitch_lvl,
        file_index=file_index,
        index_influence=index_inf,
        respiration_median_filtering=r_m_f,
        envelope_ratio=e_r,
        consonant_breath_protection=c_b_p,
        resample_sr=44100 if audio_files[0].endswith('.mp3') else 0,
    )
    time.sleep(0.1)
    result = convert_now(audio_files, random_tag, converter)
    print("Result:", result)
    return result[0]

def main():
    parser = argparse.ArgumentParser(description="Neo RVC CLI")
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")
    parser.add_argument("-aud", "--audio_files", type=str, nargs='+', required=True, help="Audio files")
    parser.add_argument("-palf0", "--pitch_alg", type=str, required=True, help="Pitch algorithm")
    parser.add_argument("-pl", "--pitch_lvl", type=int, required=True, help="Pitch level")
    parser.add_argument("-idx", "--index_inf", type=float, required=True, help="Index influence")
    parser.add_argument("-rmf", "--r_m_f", type=bool, required=True, help="Respiration median filtering")
    parser.add_argument("-er", "--e_r", type=float, required=True, help="Envelope ratio")
    parser.add_argument("-cbp", "--c_b_p", type=bool, required=True, help="Consonant breath protection")
    args = parser.parse_args()
    run(
        args.model,
        args.audio_files,
        args.pitch_alg,
        args.pitch_lvl,
        args.index_inf,
        args.r_m_f,
        args.e_r,
        args.c_b_p
    )

if __name__ == "__main__":
    main()
