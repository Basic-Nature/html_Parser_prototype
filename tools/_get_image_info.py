import os,sys
p='tools/debug_headless_output/debug_screenshot.png'
try:
    with open(p,'rb') as f:
        f.read(16)
        w=int.from_bytes(f.read(4),'big')
        h=int.from_bytes(f.read(4),'big')
    size=os.path.getsize(p)
    print(f"{w}x{h} | {size} bytes")
except Exception as e:
    print('ERROR', e)
    sys.exit(2)
