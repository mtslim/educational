import math.abs

object fixed_point {
  val tolerance = 0.0001
  def isCloseEnough(x: Double, y: Double) =
    abs((x - y) / x) / x < tolerance
  def fixedPoint(f: Double => Double)(firstGuess: Double) = {
    def iterate(guess: Double): Double = {
      val next = f(guess)
      if (isCloseEnough(guess, next)) next
      else iterate(next)
    }
    iterate(firstGuess)
  }
  fixedPoint(x => 1 + x / 2)(1) // 1.999755859375 c. 2

  // sqrt(x) is a fixed point of the function (y => x / y)
  /*
  def sqrt(x: Double) = fixedPoint(y => x / y)(1.0)
  sqrt(2) // This does not converge, it oscillates between 1 and 2
  */

  // Average damping - average successive values of the original sequence
  def sqrt(x: Double) = fixedPoint(y => (y + x / y) / 2)(1.0)
  sqrt(2)

  // Generalising the average damping function
  def averageDamp(f: Double => Double)(x: Double) = (x + f(x)) / 2

  // Using the generalised damping function
  def sqrt2(x: Double)=
    fixedPoint(averageDamp(y => x / y))(1)
  sqrt2(2)
}

class Fix() {}

