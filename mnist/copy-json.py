import os, json
import numpy as np
import random
source = 'mnist/eval/final/DJ-G/5/'
DATASET = 'mnist/original_dataset/5-LQ/'
content = os.listdir(DATASET)
con_classes = []
for file in content:
    if file.endswith('.png'):
      name = file.split('.')[0]
      # os.system(f'cp {source}{name}.json {DATASET}{name}.json')
      with open(f'{DATASET}{name}.json', 'r') as f:
          params = json.load(f)
          predictions = np.array(params['predictions'])
          _, con_class = np.argsort(-predictions)[:2]
          # print(predictions)
          # print(con_class)
          con_classes.append(con_class)
labels, counts = np.unique(con_classes, return_counts=True)
print(labels)
print(counts)

