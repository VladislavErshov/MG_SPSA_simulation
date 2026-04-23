#!/bin/bash
# Setup script for SPSA Drone Simulator virtual environment

set -e

echo "=================================="
echo "SPSA Drone Simulator Setup"
echo "=================================="

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "❌ Python not found. Please install Python 3.7 or higher."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✓ Found Python: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    echo "✓ Virtual environment created in .venv/"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install requirements
echo "⬇️  Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ Requirements installed"
else
    echo "⚠️  requirements.txt not found, installing individual packages..."
    pip install numpy matplotlib
    echo "✓ Packages installed"
fi

# Make quick_start.py executable
chmod +x examples/quick_start.py

# Run quick start test
echo ""
echo "🧪 Running initial tests..."
python examples/quick_start.py

echo ""
echo "=================================="
echo "✅ Setup complete!"
echo "=================================="
echo ""
echo "To use the simulator:"
echo "  source .venv/bin/activate"
echo "  python -m src.drone_simulator.drone_simulator"
echo "  python examples/examples.py"
echo ""
echo "To deactivate virtual environment:"
echo "  deactivate"
echo ""
