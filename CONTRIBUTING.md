# Contributing to grid_feedback_optimizer

Thank you for your interest in contributing! Contributions of all kinds are welcome: bug fixes, new features, documentation improvements, or new examples.

## 🚀 How to Get Started

1. **Fork** this repository to your GitHub account.  
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/grid_feedback_optimizer.git
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/senzhanopt/grid_feedback_optimizer.git
   ```
4. **Keep your local main branch updated**:
   ```bash
   git checkout main
   git fetch upstream
   git merge upstream/main
   git push origin main
   ```
5. **Create a new feature branch** for your work:
   ```bash
   git checkout -b feature/my-feature-name
   ```

## Code Style
- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Use type hints and meaningful variable names.
- Include docstrings for public functions and classes.

## 🧪 Testing

All contributions should include or update tests to ensure stability.

We use **pytest** for testing:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=grid_feedback_optimizer
```

- Place new tests in the `tests/` folder.
- Write both **unit tests** (for isolated components) and **integration tests** (for algorithmic correctness).
- Ensure all tests pass locally before submitting a pull request.

## 🔀 Pull Request Process

1. **Rebase or merge** your feature branch on top of the latest main from upstream:
   ```bash
   git fetch upstream
   git rebase upstream/main   # preferred for a clean linear history
   # or
   git merge upstream/main    # if you prefer merge commits
   ```
2. **Push your feature branch** to your fork:
   ```bash
   git push origin feature/my-feature-name --force-with-lease
   ```
3. **Open a Pull Request** on GitHub from your feature branch to `upstream/main`. Include:
   - A clear title (e.g., `feat: add support for primal-dual algorithm`)
   - A concise summary of what your PR does
   - References to related issues if applicable
4. A maintainer will review your PR and may request changes before merging.

## Communication
- Use [GitHub Issues](../../issues) for bugs or feature requests.
- Be respectful and constructive in discussions.

---
Maintained by [@senzhanopt](https://github.com/senzhanopt) — Licensed under MIT.
