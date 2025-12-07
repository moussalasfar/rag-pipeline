# GitHub Setup Guide

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `rag-pipeline` (or `stage_proj`)
3. Description: "Retrieval-Augmented Generation pipeline for Quorium AI Challenge"
4. Make it **PUBLIC** (important for the challenge)
5. Do NOT initialize with README (we have one)
6. Click "Create repository"

## Step 2: Push Local Repository to GitHub

Copy and run these commands in PowerShell:

```powershell
cd "c:\Users\acer\OneDrive\Bureau\stage_proj"

# Configure git if not done
git config user.name "Moussa LASFAR"
git config user.email "moussalasfar2000@gmail.com"

# Add remote repository (replace USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/rag-pipeline.git

# Rename branch to main if needed
git branch -M main

# Push all commits and tags
git push -u origin main
```

## Step 3: Verify on GitHub

1. Go to https://github.com/YOUR_USERNAME/rag-pipeline
2. Verify all files are there:
   - ✅ src/ folder with all modules
   - ✅ tests/ folder with all tests
   - ✅ README.md
   - ✅ QUICKSTART.md
   - ✅ Dockerfile
   - ✅ docker-compose.yml
   - ✅ .github/workflows/ (CI/CD pipelines)

3. Check "Actions" tab - CI/CD should run automatically

## Step 4: Create GitHub Release (Optional but Recommended)

```powershell
# Create a tag for version 1.0
git tag -a v1.0 -m "RAG Pipeline v1.0 - Quorium AI Challenge submission"

# Push the tag
git push origin v1.0
```

## Step 5: Email Submission to Quorium

Subject: RAG Pipeline - Quorium AI Challenge Submission

Body:
```
Dear Quorium Team,

I have completed the RAG Pipeline challenge. Here is my submission:

GitHub Repository: https://github.com/YOUR_USERNAME/rag-pipeline

The project includes:
- Clean RAG pipeline with document ingestion, retrieval, and generation
- Comprehensive test suite (20+ tests covering critical paths)
- Detailed documentation with architecture decisions and tradeoffs
- Docker support for easy deployment
- GitHub Actions CI/CD pipelines for automated testing

Key Features:
✅ Intelligent document chunking (500-char chunks with overlap)
✅ FAISS-based semantic search with OpenAI embeddings
✅ LLM-powered answer generation with context grounding
✅ Full test coverage for ingestion, retrieval, and generation
✅ CLI interface for easy usage

To run locally:
1. Clone the repository
2. Run: setup.bat (Windows) or bash setup.sh (macOS/Linux)
3. Add OPENAI_API_KEY to .env
4. Run: python src/cli.py ingest data/documents
5. Query: python src/cli.py query "your question"

All tests pass and the pipeline is production-ready.

Looking forward to the walkthrough!

Best regards,
Moussa LASFAR
```

---

## Troubleshooting

**Error: "fatal: Could not read from remote repository"**
- Make sure you replaced YOUR_USERNAME with your actual GitHub username
- Make sure the repository is created on GitHub first

**Error: "Git remote already exists"**
```powershell
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/rag-pipeline.git
```

**Want to verify the push worked?**
```powershell
git remote -v
```

You should see:
```
origin  https://github.com/YOUR_USERNAME/rag-pipeline.git (fetch)
origin  https://github.com/YOUR_USERNAME/rag-pipeline.git (push)
```

---

## Commits Ready to Push

```
4fab2b9 Add Docker support and GitHub CI/CD workflows
23b34c4 Add quick start guide for fast onboarding
75b6ccb Add comprehensive validation script for pipeline structure
f1558d1 Add integration tests, setup scripts, and environment template
a16aef2 Initial commit: RAG pipeline scaffold
```

Everything is ready! 🚀
