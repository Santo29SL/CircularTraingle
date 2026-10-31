# Run this in Kaggle when the ML pipeline is finished locally
!git clone https://github.com/Santo29SL/CircularTraingle.git -b itr2
%cd CircularTraingle
!pip install -r requirements-server.txt

import subprocess
import time
import pexpect

subprocess.Popen(["uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8000"])
time.sleep(5)

!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared

child = pexpect.spawn('./cloudflared tunnel --url http://127.0.0.1:8000', encoding='utf-8')
while True:
    line = child.readline()
    if "trycloudflare.com" in line:
        url = "https://" + line.split("https://")[1].split()[0]
        print(f"\n✅ BACKEND READY!")
        print(f"👉 CLICK HERE TO OPEN APP: https://huggingface.co/spaces/YOUR_NAME/Sketch2Shape?backend={url}\n")
        break