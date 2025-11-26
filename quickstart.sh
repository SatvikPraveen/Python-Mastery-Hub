#!/bin/bash
# Quick Start Script for Python Mastery Hub
# This script sets up the development environment and validates the project

set -e

echo "🐍 Python Mastery Hub - Quick Start Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

if [[ "$python_version" < "3.11" ]]; then
    echo -e "${RED}✗ Python 3.11 or higher required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Check if .env exists
echo "Checking environment configuration..."
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠ .env file not found${NC}"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
    echo -e "${YELLOW}⚠ Please review and update .env with your configuration${NC}"
else
    echo -e "${GREEN}✓ .env file exists${NC}"
fi
echo ""

# Install dependencies
echo "Installing dependencies with Poetry..."
poetry install --only=main > /dev/null 2>&1
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Test core imports
echo "Testing core module imports..."
poetry run python << 'PYTHON_TEST' > /dev/null 2>&1
try:
    from python_mastery_hub.core import get_module, list_modules
    modules = list_modules()
    if len(modules) == 9:
        print("✓ All 9 core modules loaded successfully")
    else:
        print(f"✗ Expected 9 modules, got {len(modules)}")
        exit(1)
except Exception as e:
    print(f"✗ Import test failed: {e}")
    exit(1)
PYTHON_TEST

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Core imports working${NC}"
else
    echo -e "${RED}✗ Core imports failed${NC}"
    exit 1
fi
echo ""

# Run basic CLI test
echo "Testing CLI interface..."
poetry run python -m python_mastery_hub.cli.main --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ CLI interface working${NC}"
else
    echo -e "${YELLOW}⚠ CLI interface test skipped${NC}"
fi
echo ""

echo "=========================================="
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review .env file and update configuration if needed"
echo "  2. Run CLI:        poetry run python-mastery-hub list"
echo "  3. Run Web server: poetry run uvicorn python_mastery_hub.web.main:app --reload"
echo "  4. Run tests:      poetry run pytest tests/"
echo ""
echo "Documentation: https://github.com/SatvikPraveen/Python-Mastery-Hub"
echo ""
