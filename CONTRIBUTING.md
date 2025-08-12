
# Contributing Guidelines

Thank you for your interest in contributing!  
We welcome all kinds of contributions — bug reports, bug fixes, feature implementations, documentation improvements, and more.

Please follow these guidelines to ensure a smooth and productive collaboration.

---

> **Note:** Before contributing, you must install **Git LFS** (per your OS) **and** enable **pre-commit hooks**.

## Install Git LFS

Ubuntu/Debian
```bash
sudo apt install git-lfs
git lfs install
```

macOS
```bash
brew install git-lfs
git lfs install
```

Windows
```bash
choco install git-lfs   # or: winget install Git.GitLFS
git lfs install
```

Or manual download:
https://git-lfs.github.com/

> Note: Enable pre-commits after cloning

---

## 🚀 Contribution Workflow

### 1. Fork the Repository

Click the **Fork** button at the top-right of the repository page to create your own copy.

### 2. Clone Your Fork

> Note: Setup SSH keys (if you haven’t already)

To push changes via SSH, you need to add your SSH key to your GitHub account.  
Follow the official guide here: **[GitHub SSH setup](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)**

```bash
git clone git@github.com:<your-username>/IsaacLab-SO_100.git
cd IsaacLab-SO_100
```

Enable pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

### 3. Add the Original Repository as Upstream

To keep your fork up-to-date with the original project:

```bash
git remote add upstream https://github.com/MuammerBay/IsaacLab-SO_100.git
```

To fetch and merge changes from upstream later:

```bash
git fetch upstream
git merge upstream/main
```

### 4. Create a Feature Branch

Always create a new branch from `main` for your work.  
Use a **clear, kebab-case** name describing your contribution:

```bash
git checkout -b adds-contributing-guide
```

> ✅ Example: `fix-login-issue`, `add-dark-mode`

---

## ✅ Commit & Pull Request Guidelines

### Commit Messages

- Use **lowercase** messages  
- Start with a **third-person verb** (e.g., `adds`, `updates`, `fixes`)  
- Keep messages **short and descriptive**

**Examples:**

```
adds contributing guide  
fixes broken link in readme  
updates config for python 3.10
```

### Pull Request Titles

- Follow the same format as commit messages, except a upper case is expected for the first character
- Example: `Adds user authentication`, `Fixes login redirect bug`

---

## 📦 Push Your Changes

When your changes are ready:

```bash
git add .
git commit -m "your commit message"
git push origin your-working-branch
```

---

## 📬 Open a Pull Request (PR)

1. Go to your fork on GitHub.
2. Click **Compare & pull request**.
3. Set the **base repository** to `MuammerBay/IsaacLab-SO_100` and base branch to `main`.
4. Add a clear title and short description.
5. Submit the pull request!

---

## ℹ️ Best Practices

- ✅ Keep PRs focused and minimal — one feature or fix per PR  
- 🚫 Avoid unrelated changes  
- 📖 Update documentation if needed  
- 💬 Ask questions in the PR if anything is unclear

---

Thank you again for contributing! 💙  
Let’s build something great together.

