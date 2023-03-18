import argparse
import csv
from pathlib import Path

def extract_precision_of_classes(filename):
  numbers = []
  numbers_temp = []
  classes = ["all", "person", "bicycle", "car", "motorcycle"]

  # Open the file and read the contents
  with open(filename, 'r') as f:
    contents = f.read()

  # Split the contents into a list of lines
  lines = contents.split('\n')

  # Iterate through each line
  for line in lines:
    # Split the line by spaces
    split_line = [s for s in line.split(' ') if s != ""]
    if len(split_line) == 0:
       continue
    # If it is a class AP
    if split_line[0] in classes:
        # third value is P
        numbers_temp.append(split_line[3])
    elif split_line[0] == "Speed:":
        numbers.append([line.split("inference/NMS/total per ")[1].split("x")[0]] + numbers_temp)
        numbers_temp = []
  # Return the list of numbers
  return numbers, [""] + classes

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("file", help="output txt file")
  args = parser.parse_args()

  # Call the function to extract the numbers
  numbers, headers = extract_precision_of_classes(args.file)

  # Open the output CSV file and write the numbers to it
  with open(f"output_classes_{Path(args.file).stem}.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(numbers)
