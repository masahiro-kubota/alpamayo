import re
import glob
import os

print("Scanning logs...")
logs = sorted(glob.glob("viz_*.log"))
print("| Temp | Avg Max Lat Dev (m) | Max Dev (m) | Status |")
print("|---|---|---|---|")

for log in logs:
    temp = log.split("_")[1].replace(".log", "")
    try:
        with open(log, 'r') as f:
            content = f.read()
    except:
        content = ""
        
    avg_match = re.search(r"Avg Max Lat Dev: ([\d\.]+) m", content)
    max_match = re.search(r"Max of all: ([\d\.]+) m", content)
    
    status = "running/unknown"
    if "SUCCESS! Debug visualization complete." in content:
        status = "✅ Success"
    elif "Traceback" in content or "Error" in content:
        status = "❌ Failed"
    elif "Processing bag" in content:
        status = "⏳ Running"
        
    avg = avg_match.group(1) if avg_match else "-"
    max_val = max_match.group(1) if max_match else "-"
    
    print(f"| {temp} | {avg} | {max_val} | {status} |")
