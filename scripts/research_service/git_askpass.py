#!/usr/bin/python3
import pathlib,sys
print('x-access-token' if 'username' in sys.argv[1].lower() else pathlib.Path('/data/openai-agent/.secrets/github.token').read_text().strip())
