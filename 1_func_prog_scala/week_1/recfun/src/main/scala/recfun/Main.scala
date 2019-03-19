package recfun

object Main {
  def main(args: Array[String]) {
    println("Pascal's Triangle")
    for (row <- 0 to 10) {
      for (col <- 0 to row)
        print(pascal(col, row) + " ")
      println()
    }
  }

  /**
   * Exercise 1
   */
    def pascal(c: Int, r: Int): Int = {
      def factorial(n: Int): Int =
        if (n == 0) 1 else n * factorial(n - 1)

      factorial(r) / (factorial(c) * factorial(r - c))
    }
  /**
   * Exercise 2
   */
    def balance(chars: List[Char]): Boolean = {
      // Use tail recursion to balance the number of parentheses
      // An accumulator is used to carry the total number of parentheses found
      def loop(acc: Int, chars: List[Char]): Int = {
        if (chars.isEmpty) acc
        else if (acc < 0) -1 // An unpaired closing parenthesis will produce a negative acc and will terminate the loop early
        else if (chars.head == '(') loop(acc + 1, chars.tail)
        else if (chars.head == ')') loop(acc - 1, chars.tail)
        else loop(acc, chars.tail)
      }
      // Only when the value 0 is returned by the loop will the balance function return true
      // If loop < 0 then at least one unpaired closing parenthesis
      // If loop > 0 then at least one unpaired opening parenthesis
      if (loop(0, chars) == 0) true
      else false
    }
  
  /**
   * Exercise 3
   */
    def countChange(money: Int, coins: List[Int]): Int = {
      def count(money: Int, coins: List[Int]): Int = {
        if (money == 0) 1
        else if (money > 0 && !coins.isEmpty)
          count(money - coins.head, coins) + count(money, coins.tail)
        else 0
      }
      if (money == 0 || coins.filter(_ > 0).isEmpty) 0 // Make sure input data are valid
      else count(money, coins.filter(_ > 0))
    }
  }
