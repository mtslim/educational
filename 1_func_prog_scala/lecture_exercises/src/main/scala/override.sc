object overrides {
  /* It is possible to redefine an existing, non-abstract definition in a subclass by using override */
}

abstract class Base {
  def foo = 1
  def bar: Int
}

class Sub extends Base {
  override def foo = 2 // redefining foo requires an override
  def bar = 3 // For methods that implement methods in the base class then override is not required
}