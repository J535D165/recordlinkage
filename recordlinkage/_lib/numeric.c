#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define R 6371
#define TO_RAD (3.1415926536 / 180)

double euclidean_dist(double x, double y)
{
	return fabs(y - x);
}

double haversine_dist(double th1, double ph1, double th2, double ph2)
{
	double dx, dy, dz;
	ph1 -= ph2;
	ph1 *= TO_RAD, th1 *= TO_RAD, th2 *= TO_RAD;
 
	dz = sin(th1) - sin(th2);
	dx = cos(ph1) * cos(th1) - cos(th2);
	dy = sin(ph1) * cos(th1);
	return asin(sqrt(dx * dx + dy * dy + dz * dz) / 2) * 2 * R;
}

double step_sim(double d, double offset, double origin)
{
	if (fabs(d - origin) <= offset)
	{
		return 1.0;
	} else 
	{
		return 0.0;
	}
}

double linear_sim(double d, double scale, double offset, double origin)
{

	double d_norm;

	// normalise the distance measure
	d_norm = fabs(d - origin);

	if (d_norm <= offset)
	{
		return 1.0;
	} 
	else if (d_norm >= offset + 2 * scale)
	{
		return 0.0;
	} 
	else 
	{
		return 1.0 - (d_norm - offset) / (2 * scale);
	}
}


double squared_sim(double d, double scale, double offset, double origin)
{

	double d_norm;

	// normalise the distance measure
	d_norm = fabs(d - origin);

	if (d_norm <= offset)
	{
		return 1.0;
	} 
	else if (d_norm >= offset + sqrt(2.0) * scale)
	{
		return 0.0;
	} 
	else 
	{
		return 1.0 - 0.5 * exp(2.0 * log((d_norm - offset)/scale));
	}
}


double exp_sim(double d, double scale, double offset, double origin)
{

	double d_norm;

	// normalise the distance measure
	d_norm = fabs(d - origin);

	if (d_norm <= offset)
	{
		return 1.0;
	} 
	else 
	{
		return pow(2.0, - (d_norm-offset) / scale);
	}
}


double gauss_sim(double d, double scale, double offset, double origin)
{

	double d_norm;

	// normalise the distance measure
	d_norm = fabs(d - origin);

	if (d_norm <= offset)
	{
		return 1.0;
	} 
	else 
	{
		return pow(2.0, - pow((d_norm-offset) / scale, 2.0));
	}
}
