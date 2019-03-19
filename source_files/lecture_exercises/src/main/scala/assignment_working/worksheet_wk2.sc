
object worksheet_wk2 {
  /**
    * We represent a set by its characteristic function, i.e.
    * its `contains` predicate.
    */
  type Set = Int => Boolean

  /**
    * Indicates whether a set contains a given element.
    */
  def contains(s: Set, elem: Int): Boolean = s(elem)

  /**
    * Returns the set of the one given element.
    */
  def singletonSet(elem: Int): Set = Set(elem)

  val x = singletonSet(1)
  contains(x, 1)
  contains(x, 0)

  /**
    * Returns the union of the two given sets,
    * the sets of all elements that are in either `s` or `t`.
    */
  def union(s: Set, t: Set): Set = i => s(i) || t(i)

  val y = union(singletonSet(0), singletonSet(1))
  contains(y, 1)
  contains(y, 2)

  /**
    * Returns the intersection of the two given sets,
    * the set of all elements that are both in `s` and `t`.
    */
  def intersect(s: Set, t: Set): Set = i => s(i) && t(i)

  val z = intersect(singletonSet(1), singletonSet(1))
  contains(z, 1)
  contains(z, 2)

  /**
    * Returns the difference of the two given sets,
    * the set of all elements of `s` that are not in `t`.
    */
  def diff(s: Set, t: Set): Set = i => s(i) && (!t(i))

  val w = diff(singletonSet(1), singletonSet(0))
  contains(w, 1)
  contains(w, 2)

  /**
    * Returns the subset of `s` for which `p` holds.
    */
  def filter(s: Set, p: Int => Boolean): Set = intersect(s, p)


  /**
    * The bounds for `forall` and `exists` are +/- 1000.
    */
  val bound = 1000

  /**
    * Returns whether all bounded integers within `s` satisfy `p`.
    */
  def forall(s: Set, p: Int => Boolean): Boolean = {
    def iter(a: Int): Boolean = {
      if (a > bound) true
      else if (contains(diff(s, p), a)) false
      else iter(a + 1)
    }
    iter(-bound)
  }

  forall(y, x)

  /**
    * Returns whether there exists a bounded integer within `s`
    * that satisfies `p`.
    */
  def exists(s: Set, p: Int => Boolean): Boolean =

  /**
    * Returns a set transformed by applying `f` to each element of `s`.
    */
  //def map(s: Set, f: Int => Int): Set = ???


}