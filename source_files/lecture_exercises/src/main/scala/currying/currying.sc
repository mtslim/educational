object currying {
  // Tail recursive sum function
  def sum_tail(f: Int => Int, a: Int, b: Int): Int = {
    def loop(a: Int, acc: Int): Int = {
      if (a > b) acc
      else loop(a + 1, acc + f(a))
    }
    loop(a, 0)
  }
  sum_tail(x => x * x, 3, 5)

  def sum(f: Int => Int)(a: Int, b: Int): Int =
    if (a > b) 0
    else f(a) + sum(f)(a + 1, b)
  sum(x => 2 * x)(2, 4) // 18

  def product(f: Int => Int)(a: Int, b: Int): Int =
    if (a > b) 1
    else f(a) * product(f)(a + 1, b)
  product(x =>  x * x)(3, 7) // 6350400

  // Write factorial function in terms of the product function
  def fact(n: Int): Int =
    if (n < 1) 1
    else product(x => x)(1, n)
  fact(4) // 24

  // General function to express both the sum function and the product function
  // We require a version of mapReduce: function f would map values in the
  // interval and reduce them by combining them
  // Must get the f parameter, combine (takes two ints and returns an int, zero (a unit value like 0 or 1)
  def mapReduce(f: Int => Int, combine: (Int, Int) => Int, zero: Int)(a: Int, b: Int): Int =
    if (a > b) zero
    else combine(f(a), mapReduce(f, combine, zero)(a + 1, b))


  // define product in terms of mapReduce
  def product2(f: Int => Int)(a: Int, b: Int): Int = mapReduce(f, (x, y) => x * y, 1)(a, b)
  product(x => x * x)(3, 7)
}