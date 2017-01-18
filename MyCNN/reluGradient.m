function g = reluGradient(z)

g = z;
g(g>0) = 1;

end