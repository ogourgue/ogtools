Point(1) = {-0.5,  0.0, 0.0, 0.01};
Point(2) = { 0.0,  0.0, 0.0, 0.01};
Point(3) = { 0.5,  0.0, 0.0, 0.01};

Circle(1) = {1, 2, 3};
Circle(2) = {3, 2, 1};

Physical Line("wall") = {1, 2};

Line Loop(3) = {2, 1};

Plane Surface(4) = {3};

Physical Surface("inside") = {4};
