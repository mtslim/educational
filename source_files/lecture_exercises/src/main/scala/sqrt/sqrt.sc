object sqrt {
  // Newton's method to find the square root of a number. Illustration of nesting functions to avoid
  // name-space pollution and to prevent users accessing these functions directly
  def abs(x:Double) = if (x < 0) -x else x

  def sqrt(x: Double) = {

    def sqrtIter(guess: Double): Double =
      if (isGoodEnough(guess)) guess
      else sqrtIter(improve(guess))

    def isGoodEnough(guess: Double) =
      abs(guess * guess - x) < 0.001 * x

    def improve(guess: Double) = (guess + x / guess) / 2

    sqrtIter(1.0)
  }

  sqrt(0.001)
  sqrt(1.0e-20)
  sqrt(1.0e20)
}