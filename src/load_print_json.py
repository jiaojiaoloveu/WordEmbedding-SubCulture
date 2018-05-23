import json
from pprint import pprint

with open('../data/github-repos/aamattos/GMF-Tooling-Visual-Editor.json') as f:
    data = json.load(f)
    pprint(data)

print '\nend'
