#!/bin/bash
# Nexa Toolkit Usage Examples

echo "🚀 Nexa Toolkit Usage Examples"
echo "================================"

# Check if toolkit is built
if [ ! -f "./nexa-toolkit" ]; then
    echo "Building Nexa Toolkit first..."
    go build -o nexa-toolkit Main.go
fi

echo ""
echo "1. Environment Check:"
echo "   ./nexa-toolkit check"
./nexa-toolkit check

echo ""
echo "2. Interactive TUI Mode (if model file exists):"
echo "   ./nexa-toolkit NexaBio_1.pt"
echo "   (This will launch the interactive interface)"

echo ""
echo "3. Direct CLI Operations:"

if [ -f "NexaBio_1.pt" ]; then
    echo ""
    echo "   Analyzing model..."
    echo "   ./nexa-toolkit NexaBio_1.pt analyze"
    ./nexa-toolkit NexaBio_1.pt analyze

    echo ""
    echo "   Converting to SafeTensors..."
    echo "   ./nexa-toolkit NexaBio_1.pt convert --format safetensors"
    # ./nexa-toolkit NexaBio_1.pt convert --format safetensors

    echo ""
    echo "   Quantizing model..."
    echo "   ./nexa-toolkit NexaBio_1.pt quantize --method fp16"
    # ./nexa-toolkit NexaBio_1.pt quantize --method fp16

    echo ""
    echo "   Benchmarking performance..."
    echo "   ./nexa-toolkit NexaBio_1.pt benchmark"
    # ./nexa-toolkit NexaBio_1.pt benchmark
else
    echo "   No model file found. Example commands:"
    echo "   ./nexa-toolkit model.pt analyze"
    echo "   ./nexa-toolkit model.pt convert --format safetensors"
    echo "   ./nexa-toolkit model.pt quantize --method fp16"
    echo "   ./nexa-toolkit model.pt benchmark"
fi

echo ""
echo "4. Advanced Usage:"
echo "   ./nexa-toolkit model.pt convert --format pytorch --output custom_name.pt"
echo "   ./nexa-toolkit model.pt quantize --method int8 --output quantized.pt"
echo "   ./nexa-toolkit model.pt analyze --verbose"

echo ""
echo "🎉 Ready to use Nexa Toolkit!"
echo "   For interactive mode: ./nexa-toolkit [model_file]"
echo "   For direct CLI:       ./nexa-toolkit [model_file] <operation> [flags]"
