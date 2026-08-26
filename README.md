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

## Training manifest

The normalized manifest includes `abe_new`. It retains original labels, adds
canonical numbered-pinyin fields, and uses `include_experiment=no` for unlabeled
recordings and the excluded syllabic `m`/`n` classes.

## Frozen-encoder classifier grid

The initial supervised experiment is a 2 x 2 x 2 grid:

- training speaker: Abe or Yue
- encoder: Chinese HuBERT Base or XLS-R 300M
- aggregation: global mean+standard-deviation or eight ordered temporal bins

Both heads predict a 411-way canonical base syllable. The tone head predicts
tones 1--4; unspecified and neutral-tone labels are masked from tone loss.
Oli is never used for training and is evaluated only for base syllables.

Install the additional dependencies into the existing environment:

```
python3 -m pip install -r requirements-training.txt
```

Then rebuild the manifest so the known 54-minute mislabeled recording is
excluded and submit the two-encoder Slurm array:

```
python3 scripts/build_manifest.py
sbatch slurm/frozen_encoder_grid.sbatch
```

Each array task extracts its frozen encoder features once, retaining both
aggregation representations, and then trains its four classifier variants.
Outputs are written under `results/frozen_grid/`; each run contains
`metrics.json`, `predictions.tsv`, and `classifier.pt`.

Validation uses one deterministically selected recording per base syllable,
provided the training speaker has at least two examples of that base. This
keeps all 411 base classes in training and represents all 411 in validation.
The earlier hash-fraction split remains available with
`--validation-strategy hash-fraction` for comparison.

Audit a deterministic sample of WebM decoding with:

```
python3 scripts/audit_webm_decode.py --sample-size 100
```

The audit compares decoded duration to the container duration and retains all
decoder warnings in `results/webm_decode_audit.tsv`.

### Checkpoint format

Classifier checkpoints use a versioned, top-level dictionary. The current
format is format 1:

```
{
    "format": 1,
    "state_dict": {...},
    "metadata": {...},
    "metrics": {...},
}
```

`metadata` contains everything needed to reconstruct and interpret the model,
including the exact Hugging Face revision, pooling architecture, state scope,
and base-syllable vocabulary. `metrics` contains learning history and measured
validation/test results. `state_scope` distinguishes complete frozen-encoder
heads from partial fine-tuning overlays. Checkpoints are written atomically so
an interrupted save does not replace a valid earlier file.

Inspect legacy checkpoints without changing them, then migrate them in place:

```
python3 scripts/migrate_checkpoints.py
python3 scripts/migrate_checkpoints.py --apply
```

The loader converts legacy format 0 checkpoints in memory, while the migration
tool rewrites them to format 1. Existing experiment checkpoints have already
been migrated without retraining.

Run inference on either a frozen or partially fine-tuned checkpoint with:

```
python3 scripts/predict_classifier.py \
    results/unfrozen_grid/hubert_abe_global/classifier.pt \
    path/to/recording.wav \
    --device cuda
```

The command loads the pinned pretrained revision, reconstructs the appropriate
aggregation and heads, validates every overlay key, and emits base, tone, joint,
top-k base, and tone-probability predictions as JSON.

### Experiment configuration

Version-controlled training defaults live in:

- `configs/frozen_classifier.json`
- `configs/unfrozen_classifier.json`

The Slurm grid scripts pass these files through `--config`. Individual command-
line options can still override a configured value for an explicit experiment.

## Development checks

Install the development tools and run the repository checks with:

```
python3 -m pip install -r requirements-dev.txt
python3 -m ruff format --check scripts tests
python3 -m ruff check scripts tests
python3 -m unittest discover -s tests -v
```

The GitHub Actions workflow in `.github/workflows/ci.yml` runs the same format,
lint, and unit-test checks for pushes and pull requests.

## Extended partial fine-tuning grid

The extended grid compares global statistics, 8 ordered bins, and 16 ordered
bins for both encoders and both training speakers. To keep the flattened
temporal representation fixed at 1,024 values, 8-bin pooling uses 128 features
per bin and 16-bin pooling uses 64 features per bin. Training uses at most 40
epochs, does not early-stop before epoch 15, and uses patience 6 afterward.
Checkpoints are selected first by validation base accuracy and then by tone
accuracy as a tie-breaker.

Submit the two parallel encoder branches with:

```
sbatch slurm/unfrozen_extended_grid.sbatch
```

The 12 run directories are written under `results/unfrozen_grid_extended/`,
leaving the original 20-epoch grid unchanged.

When I prompted Codex to analyze the results, eventually it picked up
that Yue was a native speaker and Abe was not, realizing the need for
some manual annotation.  It suggested also doing extracting F0
contours to further study these.

I also have another idea: do speech activity detection and rerun the
ASR results on just the speech audio, not the whole recording
including silence.

## Speech endpointing study

The endpointing experiment tests whether variable leading and trailing silence
harms ordered temporal pooling. The detector uses recording-adaptive RMS energy
in 20 ms frames, removes brief active runs, bridges short internal gaps, and
retains 80 ms of context on both sides of the detected speech. Conservative
safety checks retain the original recording when the proposed span is too short
or would remove more than 80% of the audio.

Run a small, speaker-balanced endpoint audit with:

```
python3 scripts/audit_speech_endpointing.py
```

This writes boundary metadata and dependency-free SVG plots under
`results/endpointing_audit/`. The full CPU cache and dependent HuBERT grid can
then be submitted with:

```
CACHE_JOB=$(sbatch --parsable slurm/cache_endpointed_audio.sbatch)
sbatch --dependency="afterok:$CACHE_JOB" \
    slurm/unfrozen_endpointed_hubert_grid.sbatch
```

The versioned endpointed cache is written to
`data/audio_16khz_endpointed.pt`. It contains trimmed waveforms, original and
detected boundaries, energy diagnostics, fallback status, and the complete
detector configuration. In the full 5,648-recording cache, median trimming was
54.8% for Abe, 54.2% for Yue, and 38.5% for Oli. The detector retained the
original audio for 201 Abe, 12 Yue, and 4 Oli recordings that failed a safety
check.

### Endpointed HuBERT results

The six-run grid uses the same 40-epoch partial-fine-tuning configuration as the
extended experiment. Results are stored under
`results/unfrozen_grid_endpointed/`; exported confusion matrices are under
`results/unfrozen_grid_endpointed_analysis/`.

| Training speaker | Pooling | Validation base | External base | External tone | External joint | Oli base |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Abe | global | 68.86% | **42.13%** | 66.22% | **27.91%** | **12.38%** |
| Abe | 8-bin | 40.63% | 24.38% | **67.74%** | 17.33% | 4.05% |
| Abe | 16-bin | 36.74% | 20.86% | 64.57% | 13.34% | 3.33% |
| Yue | global | **73.72%** | **26.21%** | 40.34% | **11.80%** | **2.62%** |
| Yue | 8-bin | 41.61% | 8.51% | **44.17%** | 3.80% | 1.67% |
| Yue | 16-bin | 35.77% | 7.35% | 42.29% | 3.83% | 1.43% |

Compared with otherwise matched untrimmed runs, the external changes in
percentage points were:

| Training speaker | Pooling | Base change | Tone change | Joint change |
| --- | --- | ---: | ---: | ---: |
| Abe | global | -0.59 | +2.29 | +1.12 |
| Abe | 8-bin | **+7.05** | **+9.22** | **+7.11** |
| Abe | 16-bin | **+7.76** | **+8.75** | **+6.23** |
| Yue | global | +6.32 | -8.30 | +1.47 |
| Yue | 8-bin | **+4.42** | **+8.37** | **+2.47** |
| Yue | 16-bin | **+4.17** | **+7.52** | **+2.43** |

Endpointing consistently improved every external metric for both ordered
temporal representations, supporting the hypothesis that silence-induced
misalignment had obscured their value. Eight bins nevertheless remained better
than 16, so boundary silence was not the sole cause of the 16-bin deficit.
Global pooling remained strongest for base and joint recognition, while the
Abe-trained endpointed 8-bin model achieved the best cross-speaker tone accuracy
(67.74%). This suggests that global and ordered representations may provide
complementary information.

Results on Oli did not improve consistently, so endpointing should not yet be
assumed to help learner speech. The Yue-trained global model also lost 8.30
points of external tone accuracy despite improving base recognition. These
speaker-dependent outcomes motivate retaining the untrimmed condition and
auditing pitch contours and realized-tone labels before treating endpointing as
a universal preprocessing requirement.

## End-to-end attention pooling study

The next experiment replaces fixed temporal aggregation with pooling learned
jointly from the base-syllable and tone objectives. All models receive the full,
untrimmed waveform. Chinese HuBERT supplies frame-level representations, its top
four encoder layers are fine-tuned, and the pooling output feeds separate
411-class base and four-class tone heads.

Three pooling architectures are compared:

- **Attentive global:** one learned frame-weight distribution produces a
  weighted mean and standard deviation, which are projected to 256 values.
- **Ordered 8-head:** eight content-sensitive, overlapping temporal heads each
  pool a 128-dimensional frame projection. Their 1,024 concatenated values are
  projected to 256.
- **Attentive combined:** attentive-global and ordered-head branches are each
  projected to 128 values, concatenated, and projected to the shared
  256-dimensional classifier representation.

Ordered heads use learnable Gaussian positional preferences initialized from
early to late in the recording. If `a[k,t]` is head `k`'s weight at relative
time `p[t]`, its empirical center is:

```
c[k] = sum_t a[k,t] * p[t]
```

The ordering loss penalizes adjacent centers that cross or approach within a
0.03 margin:

```
ordering_loss = mean_k max(0, c[k] - c[k+1] + 0.03)
```

For the diversity loss, each attention vector is L2-normalized and their
pairwise cosine-similarity matrix `S` is calculated. With `I` denoting the
identity matrix:

```
diversity_loss = mean((S - I) ** 2)
```

The diagonal contributes zero; similar off-diagonal attention patterns are
penalized. Thus, the conventionally named diversity loss penalizes homogeneous
or redundant heads in order to promote diversity. Both auxiliary terms have
weight 0.01:

```
loss = base_loss + tone_loss
       + 0.01 * diversity_loss
       + 0.01 * ordering_loss
```

### Silence-augmentation factor

Each pooling architecture is trained with and without randomized boundary
nonspeech, producing a 2-speaker x 3-architecture x 2-augmentation grid of 12
runs. In the augmented condition, independently sampled 0--500 ms regions are
added before and after every training example and regenerated each epoch. The
material mixes nonspeech from outside detected endpoints, approximately
level-matched noise, and digital zeros. Added samples are treated as valid
audio, not padding. Validation, external, and Oli recordings are unchanged.

Endpoint metadata is used only to obtain plausible augmentation material. The
trained model does not trim input or require the endpoint detector at
inference.

Run the two parallel speaker branches with:

```
sbatch slurm/unfrozen_attention_grid.sbatch
```

Configuration is stored in `configs/unfrozen_attention.json`. Outputs are under
`results/unfrozen_attention_grid/`, and exported summaries and confusion
matrices are under `results/unfrozen_attention_grid_analysis/`. Every run saves
the base-first `classifier.pt` as well as `classifier_best_tone.pt` and
`classifier_best_joint.pt`. The latter two support checkpoint-selection audits
without retraining.

### Attention results

The table reports the primary base-first checkpoint. “External” means the other
speaker, while Oli remains a separate learner evaluation with no tone labels.

| Training speaker | Pooling | Silence augmentation | Validation base | External base | External tone | External joint | Oli base |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Abe | attentive global | no | 72.26% | 26.32% | 55.64% | 14.92% | 11.19% |
| Abe | attentive global | yes | 72.75% | 32.96% | 64.57% | 19.39% | 11.43% |
| Abe | ordered 8-head | no | 68.37% | 23.62% | 61.40% | 14.86% | 11.19% |
| Abe | ordered 8-head | yes | 73.24% | 32.67% | **68.04%** | 22.09% | 12.86% |
| Abe | combined | no | **79.32%** | 24.79% | 54.29% | 12.57% | **13.81%** |
| Abe | combined | yes | 74.94% | **37.13%** | 62.57% | **24.50%** | 12.86% |
| Yue | attentive global | no | 38.20% | 15.48% | **52.40%** | 8.81% | 1.67% |
| Yue | attentive global | yes | 43.55% | 17.92% | 46.79% | 8.85% | 1.90% |
| Yue | ordered 8-head | no | 47.93% | 9.56% | 38.75% | 4.50% | 1.67% |
| Yue | ordered 8-head | yes | **52.80%** | 10.01% | 45.54% | 5.38% | 1.90% |
| Yue | combined | no | 46.23% | 11.77% | 45.83% | 5.83% | 1.90% |
| Yue | combined | yes | 52.07% | **22.04%** | 50.33% | **12.39%** | **3.81%** |

Silence augmentation produced the following external changes in percentage
points:

| Training speaker | Pooling | Base change | Tone change | Joint change |
| --- | --- | ---: | ---: | ---: |
| Abe | attentive global | +6.64 | +8.93 | +4.47 |
| Abe | ordered 8-head | +9.05 | +6.64 | +7.23 |
| Abe | combined | **+12.34** | +8.28 | **+11.93** |
| Yue | attentive global | +2.44 | -5.60 | +0.04 |
| Yue | ordered 8-head | +0.45 | +6.78 | +0.88 |
| Yue | combined | **+10.27** | +4.50 | **+6.56** |

Across the six paired comparisons, augmentation improved external base by an
average of 6.86 points, tone by 4.92, and joint accuracy by 5.52. The
Yue-trained attentive-global tone result is the main exception. The augmented
combined model offers the best end-to-end base/joint compromise, while the
Abe-trained augmented ordered model gives the highest cross-speaker tone
accuracy.

Compared with endpointed fixed 8-bin pooling, Abe-trained augmented ordered
attention improves external base from 24.38% to 32.67%, tone from 67.74% to
68.04%, and joint accuracy from 17.33% to 22.09%. This supports learned temporal
weighting beyond silence removal alone. Explicit endpointed global pooling
nevertheless remains the strongest Abe-trained base/joint model at 42.13% and
27.91%.

The attention heads remain correctly ordered, with typical empirical centers
near `0.24, 0.31, 0.39, 0.44, 0.52, 0.62, 0.72, 0.79`. Their learned widths are
approximately 0.19--0.24 of the recording, so they form a broad, overlapping
temporal basis rather than eight sharply discovered phonetic segments or a
latent hard VAD. Oli performance also remains low; the best attention result of
13.81% does not exceed the earlier untrimmed-global result of 14.29%.

Finally, selecting solely by same-speaker validation tone is not useful for the
Yue models: tone often reaches 100% in early epochs while base accuracy remains
near zero. Base-first selection remains the primary criterion, and the saved
alternative checkpoints are diagnostic rather than preferred models.
