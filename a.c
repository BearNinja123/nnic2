#include <stdio.h>

int plus(int a) {
  return a + a;
}

int mult(int a) {
  return a * a;
}

// choose                              - choose
// choose(int i)                       - is a function taking int
// *choose(int i)                      - returning a pointer
// (*choose(int i))(   )               - to a function
// (*choose(int i))(int)               - taking int
// int (*choose(int i))(int)           - returning int
int (*choose(int i))(int) {
  if (i == 0)
    return plus;
  return mult;
}

int main() {
  int (*fn0)(int joe, int tom);
  int (*fn1)(int); // fn is pointer to a int-returning int func
  int (*fn2)(int);

  fn1 = choose(0);
  fn2 = choose(1);

  printf("%d %d\n", (*fn1)(3), (*fn2)(3));

  int joe[3];
  joe = (int[]){1, 2, 4};
}
