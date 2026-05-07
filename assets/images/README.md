# Image Assets

Put experiment image files in this folder.

Recommended structure:

```text
assets/images/
  stimulus_a.png
  stimulus_b.png
  calibration/
  examples/
```

Use relative paths from the project root in JSON configs, for example:

```json
"image_path": "assets/images/stimulus_a.png"
```

Keep original source images unchanged when possible. If an image is edited for
the experiment, save it as a new file with a clear name.
