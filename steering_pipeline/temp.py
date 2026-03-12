"""import csv

with open("steering_test.csv", newline="") as csvfile, open("tester.txt", "w") as outfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        outfile.write(row["response"] + "\n")

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
plt.savefig("safety_histogram(nonesense).png")
plt.show()

safety_values = []

with open("classtest/log.csv", "r", encoding="utf-8") as f:
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
plt.savefig("safety_histogram.png")
plt.show()"""

count = 0
total = 0

with open("classtest/log.csv") as f:
    for line in f:
        parts = line.strip().split(",", 2)  # split at first two commas only
        
        if len(parts) >= 3:
            value = parts[2].split(",")[0]  # value right after second comma
            
            if value == "1":
                count += 1
            total += 1

percentage = (count / total) * 100 if total > 0 else 0
print(percentage)