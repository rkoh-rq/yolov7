import argparse
import csv
from pathlib import Path

def extract_number_after_last_equal_sign(filename):
  numbers = []
  headers = []
  # Open the file and read the contents
  with open(filename, 'r') as f:
    contents = f.read()

  # Split the contents into a list of lines
  lines = contents.split('\n')

  # Iterate through each line
  for line in lines:
    # Split the line by equal sign
    split_line = line.split(' = ')

    # If there is at least one equal sign in the line
    if len(split_line) > 1:
      split_line = split_line[-1]
      try:
        # Try to cast the last element as a float and append it to the numbers list
        numbers.append(float(split_line))
      except ValueError:
        # If it is not a valid float, just ignore it
        pass

    else:
        split_line = line.split('inference/NMS/total per ')
        # If there is at least one equal sign in the line
        if len(split_line) > 1:
            split_line = split_line[-1].split('x')[0]
            try:
                # Try to cast the last element as a float and append it to the numbers list
                headers.append(int(split_line))
            except ValueError:
                # If it is not a valid float, just ignore it
                pass
    

  # Return the list of numbers
  return numbers, headers

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("file", help="output txt file")
  args = parser.parse_args()

  # Call the function to extract the numbers
  numbers, headers = extract_number_after_last_equal_sign(args.file)

  # Open the output CSV file and write the numbers to it
  with open(f"output_acc_{Path(args.file).stem}.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    rearranged = list(zip(*[numbers[i: i+12] for i in range(0, len(numbers), 12)]))
    writer.writerows(rearranged)
