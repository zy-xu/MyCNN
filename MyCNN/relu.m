function h=relu(a)
  h = a;
  h(h<=0) = 0;
