import csv

def extract_number_after_last_equal_sign(filename):
  numbers = []

  # Open the file and read the contents
  with open(filename, 'r') as f:
    contents = f.read()

  # Split the contents into a list of lines
  lines = contents.split('\n')

  # Iterate through each line
  for line in lines:
    # Split the line by equal sign
    split_line = line.split(' ms inference/NMS/total per')

    # If there is at least one equal sign in the line
    if len(split_line) > 1:
      split_line = split_line[0].split("Speed: ")[-1]
      l1, l2, l3 = split_line.split("/")
      try:
        # Try to cast the last element as a float and append it to the numbers list
        numbers.append([float(l1), float(l2), float(l3)])
      except ValueError:
        # If it is not a valid float, just ignore it
        pass

  # Return the list of numbers
  return numbers

# Call the function to extract the numbers
numbers = extract_number_after_last_equal_sign('output_rain_1000_0.25_0.25.txt')

# Open the output CSV file and write the numbers to it
with open('output.csv', 'w', newline='') as f:
  writer = csv.writer(f)
  writer.writerows(numbers)
  # rearranged = list(zip(*[numbers[i: i+12] for i in range(0, len(numbers), 12)]))
  # writer.writerows(rearranged)
