import itertools
import random
from itertools import permutations

# Helper Function to Check a valid row
def validRows(candidate):
  for row in candidate:
    for i in range(len(row) - 1):
      if row[i] > row[i +1]:
        return False
  
  return True

# Helper Function to Check for a valid column
def validColumns(candidate):
  for col in range(max(len(row) for row in candidate)):
    column_elements = [row[col] for row in candidate if col < len(row)]
    for i in range(len(column_elements) - 1):
      if column_elements[i] > column_elements[i + 1]:
        return False
  
  return True

def is_valid_SYT(candidate):
 """"
  Check if the given candidate tableau is a valid standard Young tableau.

  Parameters:
  - candidate (Tuple[Tuple[int]]): The tableau to be checked.

  Returns:
  - bool: True if the matrix is valid, False otherwise.

  The function checks if the given matrix is a valid SYT matrix by verifying that:
  1. The elements in each column are in strictly increasing order.
  2. The elements in each row are in strictly increasing order.

  Example:
  >>> is_valid_SYT(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
  True
  >>> is_valid_SYT(((1, 2, 3), (5, 4), (6))
  False
  """
 return(validColumns(candidate) and validRows(candidate))

def reshape_perm(perm, shape):
  """
  Reshapes a permutation into a tableau based on the given shape.

  Parameters:
  - perm (Tuple[int]): The permutation to be reshaped.
  - shape (Tuple[int]): The shape of the resulting tableau as a weakly decreasing tuple of integers.

  Returns:
  - Tuple[Tuple[int]]: A tuple of tuples representing the reshaped permutation as a tableau.

  Example:
  >>> reshape_perm((1, 2, 3, 4, 5, 6), (3, 2, 1))
  ((1, 2, 3), (4, 5), (6,))
  """
  # A List we can change, to represent our tableau as we build it
  tableau = []

  #Fill the tableau
  index = 0
  for rowSize in shape:
    row = tuple(perm[index: index + rowSize])
    tableau.append(row)
    index += rowSize
  return tuple(tableau)

def SYTs(shape):
  """
  Generates SYTs (Standard Young Tableaux) of on the given shape.

  Parameters:
  - shape (Tuple[int]): The shape of the resulting SYTs as a tuple of integers.

  Returns:
  - List[Tuple[Tuple[int]]]: A list of valid SYTs generated based on the given shape.

  Example:
  >>> SYTs((2, 1))
  [((1, 2), (3,)), ((1, 3), (2,))]
  """
  # Valuses in our SYT
  n = sum(shape)
  permofVals = permutations(range(1, n +1))
  
  results = []
  for perm in permofVals:
    # Using our Helper lets generate some Tableaus
    newTab = reshape_perm(perm, shape)
    if(is_valid_SYT(newTab)):
      results.append(newTab)
  
  # This was my take on how to get unique elements from a list, make it into a set then bring it back to a list
  uniqueElem = set(results)
  return list(uniqueElem)

def random_SYT(shape):
  """
  Generates a random Standard Young Tableau (SYT) of the given shape.

  Parameters:
  - shape (Tuple[int]): The shape of the resulting SYT as a tuple of integers.

  Returns:
  - Tuple[Tuple[int]]: A random valid SYT generated based on the given shape.

  This function generates a random permutation of numbers from 1 to n+1, where n is the sum of the elements in the `shape` tuple. 
  It then reshapes the permutation into a tableau using the `reshape_perm` function. 
  If the resulting tableau is not valid, it shuffles the permutation and tries again. 
  The function continues this process until a valid SYT is found, and then returns the reshaped permutation as a tableau.

  Example:
  >>> random_SYT((2, 1))
  ((1, 2), (3,))
  """
  n = sum(shape)
  numbersList = list(range(1, n + 1))
  random.shuffle(numbersList)

  # Must Intialize before entering while loop. Otherwise is_valid wont work
  randPermuation = tuple(numbersList)
  randSYT= reshape_perm(randPermuation, shape)

  # Condition checks if its valid otherwise we keep on generating
  while(not is_valid_SYT(randSYT)):
    # Generate a random permuation and convert this list to a tuple in order to use the reshape method
    random.shuffle(numbersList)
    randPermuation = tuple(numbersList)

    # Assigns our new tableau to randSYT where it will be checked if its a valid SYT
    randSYT= reshape_perm(randPermuation, shape)
  
  return randSYT

def random_SYT_2(shape):
  """
  Generates a random Standard Young Tableau (SYT) of the given shape.

  Parameters:
  - shape (Tuple[int]): The shape of the resulting SYT as a tuple of integers.

  Returns:
  - Tuple[Tuple[int]]: A random valid SYT generated based on the given shape.

  The function generates a random SYT by starting off with the all zeroes tableau and greedily filling in the numbers from 1 to n. 
  The greedy generation is repeated until a valid SYT is produced.

  Example:
  >>> random_SYT_2((2, 1))
  ((1, 2), (3,))
  """
  n = sum(shape)
  tableau = [[0] * col for col in shape]
  numbers = list(range(1, n + 1))
  
  while numbers:
    num = random.choice(numbers)
    numbers.remove(num)
    
    # Trying to place num in the tableau
    isPlaced = False
    for r in range(len(shape)):
      for c in range(shape[r]):
        # If a Spot is not filled
        if tableau[r][c] == 0:
          #Put the number in that spot and check if it works
          tableau[r][c] = num
          if is_valid_SYT(tuple(tuple(row) for row in tableau)):
            isPlaced = True
            break
          else:
            # Reset if invalid
            tableau[r][c] = 0
      if isPlaced:
        break
        
    if not isPlaced:
      #Reset and retry if unable to place num
      tableau = [[0] * col for col in shape]
      numbers = list(range(1, n + 1))  # Reset numbers list
    
  return tuple(tuple(row) for row in tableau)