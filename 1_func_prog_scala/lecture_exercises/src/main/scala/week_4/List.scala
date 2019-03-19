package week_4

import java.util.NoSuchElementException

// Base trait
trait List[T] {
  def isEmpty: Boolean
  def head: T
  def tail: List[T]
}
// Implementation class
class Cons[T](val head: T, val tail: List[T]) extends List[T] { // head and tail are implemented here
  def isEmpty = false // In a con cell, isEmpty is always false because con cells are never empty
}

class Nil[T] extends List[T] {
  def isEmpty: Boolean = true
  def head: Nothing = throw NoSuchElementException("Nil.head")
  def tail: Nothing = throw NoSuchElementException("Nil.tail")
}