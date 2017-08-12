
// numeric distance functions
double euclidean_dist(double x, double y);
double haversine_dist(double th1, double ph1, double th2, double ph2);

// numeric similarity functions
double step_sim(double d, double offset, double origin);
double linear_sim(double d, double scale, double offset, double origin);
double squared_sim(double d, double scale, double offset, double origin);
double exp_sim(double d, double scale, double offset, double origin);
double gauss_sim(double d, double scale, double offset, double origin);
