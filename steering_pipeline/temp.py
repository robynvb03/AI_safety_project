import csv

with open("steering_test.csv", newline="") as csvfile, open("tester.txt", "w") as outfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        outfile.write(row["response"] + "\n")
