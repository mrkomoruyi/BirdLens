def read_txt_to_list(filename="data.txt"):
  """
  Reads a text file line by line and returns a list of strings.

  Each line from the file becomes an item in the list.
  Newline characters are stripped from the end of each line.

  Args:
    filename (str): The name of the file to read.

  Returns:
    list: A list of strings, or an empty list if the file is not found
          or an error occurs.
  """
  data_list = []
  try:
    with open(filename, 'r') as f:
      # Read all lines, strip trailing whitespace/newlines from each
      data_list = [line.rstrip() for line in f]
    print(f"Successfully read list from {filename}")
    return data_list
  except FileNotFoundError:
    print(f"Error: The file {filename} was not found.")
    return []
  except IOError as e:
    print(f"Error reading from file {filename}: {e}")
    return []