#!/usr/bin/env python3
"""
GitHub Repository Setup Script for 0DTE Commander Backtesting Engine
====================================================================

This script helps you create and push your repository to GitHub.

Prerequisites:
1. Git installed (✓ Done)
2. GitHub account
3. GitHub CLI (gh) installed, OR manual GitHub repository creation

Usage:
1. Install GitHub CLI: winget install --id GitHub.cli
2. Run: python github_setup.py
3. Or follow manual steps below
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and return the result."""
    print(f"\n🔧 {description}")
    print(f"Command: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def check_gh_cli():
    """Check if GitHub CLI is installed."""
    try:
        result = subprocess.run(["gh", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def setup_github_repo():
    """Guide through GitHub repository setup."""
    print("🚀 GitHub Repository Setup for 0DTE Commander Backtesting Engine")
    print("=" * 60)

    # Check if gh CLI is available
    if check_gh_cli():
        print("✅ GitHub CLI detected!")
        print("\nOption 1: Use GitHub CLI (Recommended)")

        repo_name = "0DTE-Commander-Backtesting-Engine"
        description = "A robust vectorized backtesting framework for high-frequency trading strategies based on Velocity, Drift, and Volatility Physics"

        print(f"Repository name: {repo_name}")
        print(f"Description: {description}")

        create_cmd = f'gh repo create {repo_name} --description "{description}" --public --source=. --remote=origin --push'
        if run_command(create_cmd, "Creating GitHub repository and pushing code"):
            print("\n🎉 Repository created and code pushed successfully!")
            print(f"Visit: https://github.com/YOUR_USERNAME/{repo_name}")
            return True
    else:
        print("❌ GitHub CLI not found.")
        print("\nOption 2: Manual GitHub Setup (Alternative)")
        print_manual_steps()
        return False

def print_manual_steps():
    """Print manual steps for GitHub setup."""
    print("\n📋 Manual GitHub Repository Creation Steps:")
    print("-" * 50)
    print("1. Go to https://github.com and sign in")
    print("2. Click the '+' icon → 'New repository'")
    print("3. Repository name: 0DTE-Commander-Backtesting-Engine")
    print("4. Description: A robust vectorized backtesting framework for high-frequency trading strategies")
    print("5. Make it Public (or Private if preferred)")
    print("6. DO NOT initialize with README, .gitignore, or license")
    print("7. Click 'Create repository'")
    print("\n8. Copy the repository URL from the setup page")
    print("9. Run these commands in your terminal:")
    print()
    print("   # Replace YOUR_USERNAME with your GitHub username")
    print("   git remote add origin https://github.com/YOUR_USERNAME/0DTE-Commander-Backtesting-Engine.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    print()
    print("10. Done! Your code is now on GitHub 🚀")

def main():
    """Main setup function."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Check if we're in a git repository
    if not os.path.exists('.git'):
        print("❌ Not a Git repository. Please run 'git init' first.")
        sys.exit(1)

    setup_github_repo()

if __name__ == "__main__":
    main()