#!/bin/bash
# Generate 10 visualizations at different timestamps

cd /workspace/alpamayo
source ar1_venv/bin/activate

echo "Generating 10 visualizations at different time ratios..."
echo "=========================================="

for i in 1 2 3 4 5 6 7 8 9 10; do
    ratio="0.$i"
    if [ $i -eq 10 ]; then
        ratio="1.0"
    fi
    
    output="viz_ratio_${ratio}.png"
    
    echo ""
    echo "[$i/10] Generating visualization at ratio=$ratio"
    echo "Output: $output"
    
    python run_final_viz.py --ratio $ratio --output $output
    
    if [ $? -eq 0 ]; then
        echo "✓ Success: $output"
    else
        echo "✗ Failed at ratio=$ratio"
    fi
done

echo ""
echo "=========================================="
echo "All visualizations complete!"
echo "Generated files:"
ls -lh viz_ratio_*.png 2>/dev/null | awk '{print "  " $9 " - " $5}'
