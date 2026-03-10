import csv

"""with open("steering_test.csv", newline="") as csvfile, open("tester.txt", "w") as outfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        outfile.write(row["response"] + "\n")"""

import matplotlib.pyplot as plt

safety_values = []

with open("log.csv", "r", encoding="utf-8") as f:
    next(f)  # skip header
    for line in f:
        parts = line.split(",", 3)  # only split first 3 commas
        try:
            safety_values.append(float(parts[1]))
        except (IndexError, ValueError):
            pass

plt.hist(safety_values, bins=30, edgecolor="black", linewidth=1)

plt.xlabel("Safety")
plt.ylabel("Frequency")
plt.title("Histogram of Safety Scores")
plt.show()