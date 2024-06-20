import itertools
#
def parenthesizations(n):
  """
  Returns a list of all possible parenthesizations of length n.

  Parameters:
    n (int): The length of the parenthesizations.

  Returns:
    A list of strings, where each inner string represents a valid parenthesization of length n.
  
  Example:
  >>> parenthesizations(3)
  {'((()))', '(()())', '(())()', '()(())', '()()()'}
  """
  if n == 0:
    return {""}
  else:
    results = set()
    def createParenth(curr_string, open_count, close_count):
      # Base Case where if we have equal brackets then we can say its closed
      if (open_count == n and close_count == n):
        results.add(curr_string)
        return
      
      if open_count < n :
        createParenth(curr_string + "(", open_count + 1, close_count)
      
      if close_count < open_count:
        createParenth(curr_string + ")", open_count, close_count + 1)
    createParenth("", 0, 0)
    return results

  

def product_orders(n):
  """
  Returns a list of all possible ways to multiply of n elements.

  Parameters:
    n (int): The number of elements multiplied.

  Returns:
    A set of strings where each string represents a way to multiply n elements.
  
  Example:
  >>> product_orders(4)
  {'((?*?)*?)*?', '(?*(?*?))*?', '(?*?)*(?*?)', '?*((?*?)*?)', '?*(?*(?*?))'}
  """

  if n == 0:
    return {""}
  elif n == 1:
    return {"?"}
  elif n == 2:
    return {"?*?"}
  else:
    results = set()
    for k in range(1, n):
      leftOrders = product_orders(k)
      rightOrders = product_orders(n - k)
      for left in leftOrders:
        for right in rightOrders:
          results.add(f'({left}*{right})')
    return results
  


# Helper Function to check if a subsequence follows a 2 - 3 - 1 ordering
def permCondition(subsequence):
  # Check if the first digit is the second largest
  if subsequence[0] == sorted(subsequence)[-2]:
    #Check if the second digit is the largest
    if subsequence[1] == max(subsequence):
      # Check if the last digit is the smallest
      if subsequence[2] == min(subsequence):
        return False
  return True


def permutations_avoiding_231(n):
  """
  Returns a list of permutations of length n avoiding the pattern 2-3-1.
  
  Parameters:
    n (int): The length of the permutation.
  
  Returns:
    A list of permutations of length n that do not contain the pattern 2-3-1.
  
  Example:
  >>> permutations_avoiding_231(4)
  {(1, 2, 3, 4), (1, 2, 4, 3), (1, 3, 2, 4), (1, 4, 2, 3), (1, 4, 3, 2), (2, 1, 3, 4), (2, 1, 4, 3), (3, 1, 2, 4), (3, 2, 1, 4), (4, 1, 2, 3), (4, 1, 3, 2), (4, 2, 1, 3), (4, 3, 1, 2), (4, 3, 2, 1)}
  """
  if n < 3:
    return set(itertools.permutations(range(1, n+1)))
  else:
    # Collect Valid Permutations
    validPerms = set()
    
    # Go through every permutation
    for perm in itertools.permutations(range(1, n+1)):
      # Check if they have 231 pattern within any subsequence
      if permCondition(perm):
        # If a permuation comes up clean add it to our list
        validPerms.add(perm)
    return validPerms
    

def triangulations(n):
  """
  Returns a list of all possible triangulations of an n-sided polygon. A triangulation
  is represented as a list of internal edges. Vertices are labeled 0 through n-1 clockwise.

  Parameters:
    n (int): The number of sides of the polygon.

  Returns:
    A set of tuple of pairs, where each pair represents an internal edge in the triangulation.
  
  Example:
  >>> triangulations(3) 
  {((0, 3), (1, 3)), ((1, 4), (2, 4)), ((1, 3), (1, 4)), ((0, 2), (2, 4)), ((0, 2), (0, 3))}
  """
  if n < 3:
    return set()
  elif n == 3:
    return {tuple()}
  else:
    result = set()
    for k in range(1, n-1):
      left_triangulations = triangulations(k)
      right_triangulations = triangulations(n - k)
      for left in left_triangulations:
        for right in right_triangulations:
          result.add((left[0], right[0]))
          result.add((left[1], right[0]))
    return result
   