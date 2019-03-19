object rationals {
  val x = new Rational(1, 3)
  val y = new Rational(5, 7)
  val z = new Rational(3, 2)
  val w = new Rational(2)
//  val strange = new Rational(1, 0)

  //val a = x.add(y)
  val a = x + y
  //val b = x.add(y).sub(z)
  val b = x + y - z
  //val c = x.less(y)
  val c = x < y
  val d = x.max(y)
//  val e = strange.add(strange)



}

println(rationals.x.numer)
println(rationals.x.denom)
println(rationals.w)

println(rationals.a)
println(rationals.b)
println(rationals.c)
println(rationals.d)

println



class Rational(x: Int, y: Int) {
  require(y != 0, "Denominator must be non-zero")

  def this(x: Int) = this(x, 1)

  // private members can only be accessed from inside the Rational class
  private def gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)
  def numer = x
  def denom = y

  // We can use the < symbol instead of .less()
  def <(that: Rational) = numer * that.denom < that.numer * denom
  //def less(that: Rational) = numer * that.denom < that.numer * denom
  def max(that: Rational) = if (this < that) that else this
  //def max(that: Rational) = if (this.less(that)) that else this

  //def add(that: Rational) =
  def + (that: Rational) =
    new Rational(
      numer * that.denom + that.numer * denom,
      denom * that.denom
    )

  // We require unary to allow it to be used as a prefix operator
  //def neg() = new Rational(-numer, denom)
  def unary_- : Rational = new Rational(-numer, denom) // Note: space required before the colon

  //def sub(that: Rational) = add(that.neg)
  def -(that: Rational) = this + -that

  override def toString = {
    val g = gcd(numer, denom)
    numer / g + "/" + denom / g}

}

// Fully parenthesized version of
// a + b ^? c ?^ d less a ==> b | c
// ((a + b) ^? (c ?^ d)) - ((a ==> b) | c)