# Recording File Names

Uploaded recording files are stored under Django's `MEDIA_ROOT`, which is
currently `data/media` in development.

The relative path format is:

```text
recordings/<participant-public-id>/<session-public-id>/<stimulus-index>_<attempt-number>_<stimulus-label>_<recorded-at-utc>_<recording-id>.<extension>
```

Example:

```text
recordings/participant_abc123/2d2a7e68-4c2f-4f69-b7a6-7b7d13c2e7bb/0012_02_ma3_20260731T143022Z_7f7c9ec4-7b38-4a9e-9152-9c5a7a8e4810.webm
```

Fields:

- `<participant-public-id>`: the participant's stable public id.
- `<session-public-id>`: the recording session UUID.
- `<stimulus-index>`: the 1-based prompt number inside that recording session,
  zero-padded to four digits, such as `0012`.
- `<attempt-number>`: the attempt number for that prompt, zero-padded to two
  digits, such as `02`.
- `<stimulus-label>`: the tone-bearing stimulus id, such as `ma3`; for
  tone-unspecified prompts this is the plain base syllable, such as `ma`.
- `<recorded-at-utc>`: UTC timestamp from `RecordingAttempt.recorded_at`, formatted
  as `YYYYMMDDTHHMMSSZ`.
- `<recording-id>`: the recording attempt UUID.
- `<extension>`: the lowercase extension from the uploaded filename, usually
  `.webm`.

Both `raw_audio` and `wav_audio`, if present, use this same naming scheme. Retry,
timeout, speaker-rejected, and aborted attempts may have database rows without an
audio file.
