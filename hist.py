import csv
import cv2
import numpy as np

lines = []
with open ('data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements =[]

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data1/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.resize(image, None, fx=0.5, fy=0.5)
    image = image / 255.0 - 0.5;

    measurement = float(line[3])

    if not(abs(measurement) < 0.1):
        images.append(image)
        measurements.append(measurement)
import matplotlib.pyplot as plt
import numpy as np

# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api


print(min(measurements))
print(max(measurements))
plt.hist(measurements)

plt.show()

