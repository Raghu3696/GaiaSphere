# Contributing to GaiaSphere

Thank you for your interest in GaiaSphere! Contributions from the remote sensing, ML, and climate science communities are especially valued.

## Areas for Contribution

- **New data loaders** — MODIS, Landsat, Planet, GOES-16
- **Model improvements** — Pre-training objectives (MAE, contrastive), efficiency (FlashAttention)
- **New tasks** — Sea level rise, urban heat islands, glacial retreat
- **Benchmarks** — Evaluation on new geographic regions and disaster events
- **Documentation** — Tutorials, dataset guides, architecture diagrams

## Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Run tests (`pytest tests/`)
4. Format code (`black src/`)
5. Open a pull request with a clear description

## Code Style

- Python 3.10+, type hints throughout
- `black` for formatting, `ruff` for linting
- Docstrings in Google style
- Unit tests for all new model components

## Questions

Open an issue with the `question` label — happy to discuss!
