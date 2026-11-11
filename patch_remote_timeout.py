from pathlib import Path

path = Path(
    "/home/wuzewen/Projects/Omni/releases/OmniFlow-exp-mobilegpt-5250817c/"
    "src/integrations/official_forward.py"
)
source = path.read_text()
old = "MOBILEGPT_HANDSHAKE_TIMEOUT_SEC = 60.0"
new = '''MOBILEGPT_HANDSHAKE_TIMEOUT_SEC = max(
    60.0,
    float(os.environ.get("MOBILEGPT_HANDSHAKE_TIMEOUT_SEC", "180")),
)'''
assert source.count(old) == 1
path.write_text(source.replace(old, new, 1))
