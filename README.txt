# Mandarin tones data an analysis

## current plan:
- try zero shot ASR followed by character to pinyin conversion
  - one issue with this is character to pinyin isn't one-to-one
    - remedy: consider main pinyin and alternates in a strict and lenient
      setting
  - asr might emit multiple character, latin text, etc
    - remedy: need to normalize things like traditional/simplified, remove
      punct and whitespace, etc
  - another issue is rare or non existent syllables, ie syllables with no
    real character
    - remedy: stratify the results into common lexical syllable tone combos,
      rare syllables, and non-lexical categories
  - recommendations: whisper large-v3, whisper large-v3-turbo, large-v3 with
    explicit syllable prompt, then maybe one other chinese specific model
- try a mfcc/log-mel features plus a conventional classifier
- try frozen pretrained speech embeddings with separate syllable and tone heads,
  possibly supplemented by pitch features

## future issues
- speaker independent training/evaluation
- treating syllable and tone differently?
- what to do with the unspecified tones data
- 

## initial Whisper large-v3 experiment

Build the normalized manifest (this requires only the Python standard library):

```
python3 scripts/build_manifest.py
```

Create an environment with a CUDA-compatible PyTorch installation, then install
the ASR dependency:

```
python3 -m pip install -r requirements-asr.txt
```

Run a small smoke test before submitting the full job. The first invocation
downloads the large-v3 weights into Whisper's model cache:

```
python3 scripts/transcribe_whisper.py --device cuda --limit 10
```

The runner appends one JSON object per recording and skips paths already found
in the output, so an interrupted job can be restarted safely. For a cluster run,
customize the environment setup and resource directives in
`slurm/whisper_large_v3.sbatch`, then submit it with `sbatch`.
      
