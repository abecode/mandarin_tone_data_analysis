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

Model weights are cached under `models/whisper`. The runner appends one JSON
object per recording and skips paths already found in the output, so an
interrupted job can be restarted safely. For a cluster run,
customize the environment setup and resource directives in
`slurm/whisper_large_v3.sbatch`, then submit it with `sbatch`.

Project-local static FFmpeg executables are stored under `models/linux`; the
transcription runner adds that directory to `PATH`. This avoids requiring a
system FFmpeg installation on every compute node.

Export successful and failed results to a segment-level TSV (one row per
Whisper segment; blank segment fields for empty/error results):

```
python3 scripts/export_whisper_tsv.py results/whisper_large_v3.jsonl \
  --output results/whisper_large_v3.tsv
```
      
some findings:

Several strong patterns stand out.

  1. Whisper is much better at the base syllable than the tone.

  Using preliminary pypinyin conversion on the 2,122 tone-labeled recordings:

  - Exact base syllable: approximately 35.0%
  - Exact syllable plus tone: approximately 13.1%
  - Among outputs containing exactly one Chinese syllable:
      - Base syllable: 48.9%
      - Syllable plus tone: 18.3%
      - Correct tone conditional on a correct base: 37.3%

  These are provisional because default character-to-pinyin conversion does not fully resolve polyphonic characters, and this calculation excludes useful Latin-script
  outputs.

  Examples where Whisper heard the base but selected a character with another tone include:

  chu4 → 出  chu1
  hai2 → 嗨  hai1
  mian3 → 麵  mian4
  nian3 → 年  nian2
  dui2 → 对  dui4
  wo2 → 我  wo3

  This supports evaluating base and tone separately.

  2. Whisper strongly favors familiar lexical outputs.

  The most frequent outputs include:

  好       50
  Ciao     32
  1        30
  盼       29
  嗯       29
  對       26
  全       22
  是       22
  哇       22
  烦       22

  Whisper often maps acoustically similar unfamiliar syllables to common characters, conversational expressions, Latin text, or digits. Other recurring outputs include
  拜拜, 加油, OK, and 所以.

  The pang4 repetitions are especially striking:

 The pang4 repetitions are especially striking:

  - 28 became 盼 (pan4)
  - 22 became 烦 (fan2)

  So Whisper is highly consistent on those recordings but appears uncertain about the final nasal and sometimes the initial.

  3. Non-Chinese output is common.

  Across all 2,889 recordings:

  - 2,165 produced Chinese-character output
  - 520 produced Latin-script output
  - 101 produced digits
  - 93 were empty
  - A few mixed scripts

  There were only 987 unique transcriptions, showing substantial collapse toward a smaller collection of familiar outputs.

  4. Tone 4 was rarely rejected as silence.

  Empty-output rates were:

  Tone 1: 6.3%
  Tone 2: 2.0%
  Tone 3: 1.8%
  Tone 4: 0.2%
  Tone 5: 11.1%  (only 18 recordings)
  Unspecified: 5.0%

  Tone 4’s sharp falling contour or stronger articulation may make it easier for Whisper’s speech detector to accept, although speaker/session confounding means we
  should not yet treat that as a linguistic conclusion.

  Interestingly, among single-Chinese-syllable outputs, tone-4 recordings had lower base-syllable agreement than tones 1–3:

  Tone 1: 48.6%
  Tone 2: 53.3%
  Tone 3: 55.5%
  Tone 4: 39.3%

5. There is one major data-contamination problem.

  This file is labeled ai1 but lasts 3,253 seconds—over 54 minutes:

  raw/mandarin-tone-recordings/audio_fixed_abe/abe/s001/0012_ai1_ai_a1f18643.wav

  It contains conversational speech and generated 469 Whisper segments. It is responsible for a large portion of the extra rows in the TSV and should be excluded from
  the isolated-syllable experiment.

  All other files are under seven seconds, so this is a clear outlier rather than merely a long pronunciation.

  6. A few short recordings trigger repetition loops.

  Examples include:

  - A 1.32-second yan recording decoded as 123456...30, extending to 30 seconds
  - A 3.42-second zhong recording decoded as the same phrase repeatedly, extending to 28 seconds
  - Several outputs containing repeated subtitle boilerplate or punctuation

  Seven recordings contain inferred segment endpoints beyond ten seconds. These should be marked as hallucination/repetition failures.

  Crucially, some repetition loops have extremely high average log probabilities. Therefore, avg_logprob by itself cannot be interpreted as confidence that the
  transcription is sensible. Output duration relative to audio duration, repetition, and output length must also be checked.

  7. The result is promising as an ASR baseline.

  The headline is not simply “Whisper performs poorly.” A more informative interpretation is:

  - It recovers the base syllable moderately often when it emits one Chinese syllable.
  - Tone recovery through character choice is substantially weaker.
  - It has a strong lexical prior that is poorly matched to rare or nonlexical isolated syllables.
  - It sometimes substitutes Latin text, numbers, or familiar phrases.
  - Short isolated audio can trigger both no-speech decisions and decoder hallucinations.

  Before calculating final metrics, I recommend excluding the 54-minute contaminated file, flagging repetition failures, normalizing the pinyin spelling conventions, and
  implementing strict versus polyphonic/lenient character-to-pinyin scoring.
