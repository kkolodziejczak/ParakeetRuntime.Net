"""Builds the benchmark corpus that Run-Matrix.ps1 reads, from FLEURS dev. See docs/benchmarks-howto.md."""
import array
import csv
import os
import struct
import sys
import tarfile
import urllib.request
import wave

LANGUAGES = ["pl_pl", "en_us"]
CLIPS = 150
SILENCE_SECONDS = 0.5
RATE = 16000
FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
DEFAULT_ROOT = os.path.join(os.environ.get("USERPROFILE", os.path.expanduser("~")), ".councils", "local-parakeet-engine", "fable", "corpus")


def fetch(url, path):
    if not os.path.exists(path):
        print("download", url, flush=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return [row for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if len(row) >= 3]


def read_pcm16(path):
    data = open(path, "rb").read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a WAV file")
    pos, fmt, samples = 12, None, None
    while pos + 8 <= len(data):
        chunk, size = data[pos:pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8:pos + 8 + size]
        if chunk == b"fmt ":
            fmt = struct.unpack_from("<HHI", body)
        elif chunk == b"data":
            samples = body
        pos += 8 + size + (size & 1)
    tag, channels, rate = fmt
    if channels != 1 or rate != RATE:
        raise ValueError(f"{path} is not 16 kHz mono: {fmt}")
    if tag == 1:
        return samples
    if tag != 3:
        raise ValueError(f"{path} has WAV format {tag}")
    floats = array.array("f", samples[:len(samples) // 4 * 4])
    return array.array("h", (max(-32768, min(32767, round(x * 32767))) for x in floats)).tobytes()


def write_wav(path, pcm16):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm16)


def write_list(folder, names):
    with open(os.path.join(folder, "list.txt"), "w", encoding="utf-8") as f:
        f.writelines(os.path.join(folder, name) + "\n" for name in names)


def write_reference(folder, rows):
    with open(os.path.join(folder, "dev.tsv"), "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n").writerows(rows)


def build_joined(language_dir, clips, seconds):
    folder = os.path.join(language_dir, f"long{seconds}")
    os.makedirs(folder, exist_ok=True)
    silence = b"\0\0" * int(RATE * SILENCE_SECONDS)
    groups, current = [], []
    for clip in clips:
        current.append(clip)
        if sum(len(pcm) for _, _, pcm in current) / 2 / RATE >= seconds:
            groups.append(current)
            current = []
    names, rows = [], []
    for i, group in enumerate(groups):
        name = f"long{seconds}_{i:03}.wav"
        write_wav(os.path.join(folder, name), silence.join(pcm for _, _, pcm in group))
        names.append(name)
        rows.append([str(i), name, " ".join(text for _, text, _ in group)])
    write_reference(folder, rows)
    write_list(folder, names)
    print(f"{folder}: {len(names)} files", flush=True)


def build_language(root, language):
    language_dir = os.path.join(root, language)
    wav_dir = os.path.join(language_dir, "wav16")
    os.makedirs(wav_dir, exist_ok=True)
    fetch(f"{FLEURS}/{language}/dev.tsv", os.path.join(language_dir, "dev.tsv"))
    archive = os.path.join(language_dir, "dev.tar.gz")
    fetch(f"{FLEURS}/{language}/audio/dev.tar.gz", archive)
    rows = read_rows(os.path.join(language_dir, "dev.tsv"))[:CLIPS]
    wanted = {row[1] for row in rows}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = os.path.basename(member.name)
            if member.isfile() and name in wanted and not os.path.exists(os.path.join(wav_dir, name)):
                with tar.extractfile(member) as source, open(os.path.join(wav_dir, name), "wb") as target:
                    target.write(source.read())
    clips = [(row[1], row[2], read_pcm16(os.path.join(wav_dir, row[1]))) for row in rows]
    for name, _, pcm in clips:
        write_wav(os.path.join(wav_dir, name), pcm)
    write_list(wav_dir, [row[1] for row in rows])
    print(f"{wav_dir}: {len(clips)} clips, {sum(len(pcm) for _, _, pcm in clips) / 2 / RATE:.0f} s", flush=True)
    for seconds in (60, 120):
        build_joined(language_dir, clips, seconds)


if __name__ == "__main__":
    corpus_root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    for language in LANGUAGES:
        build_language(corpus_root, language)
