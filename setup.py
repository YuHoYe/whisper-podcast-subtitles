#@markdown **配置Whisper/Setup Whisper** 🏗️

!pip install requests beautifulsoup4
!pip install git+https://github.com/openai/whisper.git

import torch
import sys

device = torch.device('cuda:0')
print('Using device:', device, file=sys.stderr)

print('Whisper installed，please execute next cell')