object intsets {
  val t1 = new NonEmpty(3, Empty, Empty)
  val t2 = t1 incl 4
}

abstract class IntSet {
  /* Abstract classes contain members which are missing an implementation/no body (in this case: incl and contains)
   * consequently no instances of an abstract class can be created with the operator new */
  /* The classes Empty and NonEmpty both extend the class IntSet. This implies that they conform to the type IntSet,
   * so an object of type Empty or NonEmpty can be used wherever an object of type IntSet is required */
  def incl(x: Int): IntSet
  def contains(x: Int): Boolean
  def union(other: IntSet): IntSet
}

/* There is really only a single empty IntSet. Rather than create many instances of it, we can express this case with
 * an object definition. This defines a singleton object named Empty. No other Empty instance need to be created.
  * Singleton objects are values, so Empty evaluates to itself. */

/*class Empty extends IntSet {
  def contains(x: Int): Boolean = false
  def incl(x: Int): IntSet = new NonEmpty(x, new Empty, new Empty)
  override def toString = "."
} */

object Empty extends IntSet {
  def contains(x: Int): Boolean = false
  def incl(x: Int): IntSet = new NonEmpty(x, Empty, Empty)
  override def toString = "."
  def union(other: IntSet): IntSet = other
}

class NonEmpty(elem: Int, left: IntSet, right: IntSet) extends IntSet {
  def contains(x: Int): Boolean = // Makes use of the sorted characteristic of trees
    if (x < elem) left contains x // Only need to look in the left sub tree
    else if (x > elem) right contains x // Only need to look in the right sub tree
    else true
  def incl(x: Int): IntSet =
    if (x < elem) new NonEmpty(elem, left incl x, right) // If the element is less than the current element then include it in the left sub tree
    else if (x > elem) new NonEmpty(elem, left, right incl x) // Include it in the right sub tree
    else this // Otherwise the element is already in the tree and can be returned as is
  override def toString = "{" + left + elem + right + "}"
  def union(other: IntSet): IntSet =
    ((left union right) union other) incl elem
}
/* Add a new method (union) to intSet class hierarchy and implement in the two subclasses
