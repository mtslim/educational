object factorial {
  // Non-tail recursive factorial function
  def factorial(n: Int): Int =
    if (n == 0) 1 else n * factorial(n - 1)

  factorial(4)
  // --> 4 * factorial(3)
  // --> 4 * 3 * factorial(2)
  // --> 4 * 3 * 2 * factorial(1)
  // --> 4 * 3 * 2 * 1 * factorial(0)
  // --> 4 * 3 * 2 * 1 * 1
  // --> 120
  // In the reduction sequence there is a build up as elements are added to our expression
  // and it gets bigger and bigger


  // Tail recursive factorial function (can be used to avoid deep recursive chains)
  // Takes two parameters, n and acc (accumulator, initial value of 1)
  def factorial_tail(n: Int): Int = {
    def loop(acc: Int, n: Int): Int =
      if (n == 0) acc
      else loop(acc * n, n -1)
    loop(1, n)
  }
  factorial_tail(4)
  factorial_tail(20)
}