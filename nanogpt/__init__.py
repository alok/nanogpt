# %%

from pathlib import Path
import urllib.request

INPUT_FILE = Path("input.txt")
data = urllib.request.urlretrieve(
    url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    filename=INPUT_FILE,
)
data = INPUT_FILE.read_text()

chars = sorted(set(data))
text = ''.join(chars)
