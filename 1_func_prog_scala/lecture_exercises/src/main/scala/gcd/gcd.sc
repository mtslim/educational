object gcd {
  // Find the greatest common divisor
  def gcd(a: Int, b: Int): Int =
    if (b == 0) a else gcd(b, a % b)

  gcd(14, 21)
  // --> gcd(21, 14)
  // --> gcd(14, 7)
  // --> gcd(7, 0)
  // --> if (0 == 0) 7 else gcd(0, 7 % 0)
  // --> 7
  // Calls itself and translates into a re-writing sequence of constant size
}
