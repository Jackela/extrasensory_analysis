# AI maintenance entry

Read README.md, report/data/README.md, the manifest, and the relevant configuration/source before editing. Keep scientific conclusions distinct from descriptive checks of committed exports. Preserve raw data and historical report artifacts.

Use Markdown for explanations and JSON/YAML for precise fields. Run `python tools/verify_report_data.py` and the report-evidence unittest for changes to published numbers or paths. Regenerate the manifest with `--write` only after reviewing actual CSV changes. A full research rerun needs the original dataset, JIDT, and declared resources; do not infer it from the export checks. Keep source, checks and release status explicit.
