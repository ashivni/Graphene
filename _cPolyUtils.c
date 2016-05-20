#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static char module_docstring[] =  "This module provides python interface to various utility functions written in C for the grainBdr class.";
static char vorCentroid_docstring[] =  "Input: start, end, regions, point_region, points_x, points_y, vertices_x, vertices_y, out_x, out_y";
static char closest_docstring[] =  "Input: p1, p2. For each point in p1, returns the index of the point in p2 that its closest to.";
static char selfClosest_docstring[] =  "Input: p1. For each point in p1, returns the index of the point in p1 that its closest to.";

static PyObject * cPolyUtils_vorCentroid(PyObject *self, PyObject *args);
static PyObject * cPolyUtils_closest(PyObject *self, PyObject *args);
static PyObject * cPolyUtils_selfClosest(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
	{"vorCentroid", cPolyUtils_vorCentroid, METH_VARARGS, vorCentroid_docstring},
	{"closest", cPolyUtils_closest, METH_VARARGS, closest_docstring},
	{"selfClosest", cPolyUtils_selfClosest, METH_VARARGS, selfClosest_docstring},
	{NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC init_cPolyUtils(void){
	PyObject *m = Py_InitModule3("_cPolyUtils", module_methods, module_docstring);
	if (m == NULL)
	return;

	/* Load `numpy` functionality. */
	import_array();
}



static PyObject *cPolyUtils_selfClosest(PyObject *self, PyObject *args)
{
	double lx=0.f, ly=0.f;
	// all except regions are ndarrays, regions is a list of lists.
	PyObject *p1_x, *p1_y, *ind_out, *dist_out;
	PyObject *p1_x_arr, *p1_y_arr, *ind_out_arr, *dist_out_arr;
	
	// Get list from input arguments
	if (! PyArg_ParseTuple( args, "ddO!O!O!O!", &lx, &ly, &PyArray_Type, &p1_x, &PyArray_Type, &p1_y, &PyArray_Type, &ind_out, &PyArray_Type, &dist_out )) return NULL;

	//---- Convert, check
	p1_x_arr = PyArray_FROM_OTF(p1_x, NPY_DOUBLE, NPY_IN_ARRAY);	
	if (p1_x_arr == NULL)
	{
		PyErr_SetString(PyExc_RuntimeError,"Not able to convert p1_x to NPY_IN_ARRAY");
	}

	p1_y_arr = PyArray_FROM_OTF(p1_y, NPY_DOUBLE, NPY_IN_ARRAY);	
	if (p1_y_arr == NULL)
	{
		Py_DECREF(p1_x_arr);
		PyErr_SetString(PyExc_RuntimeError,"Not able to convert p1_y to NPY_IN_ARRAY");
	}

	ind_out_arr = PyArray_FROM_OTF(ind_out, NPY_DOUBLE, NPY_INOUT_ARRAY);	
	if (ind_out_arr == NULL)
	{
		Py_DECREF(p1_x_arr);
		Py_DECREF(p1_y_arr);
		PyErr_SetString(PyExc_RuntimeError,"Not able to convert ind_out to NPY_INOUT_ARRAY");
	}

	dist_out_arr = PyArray_FROM_OTF(dist_out, NPY_DOUBLE, NPY_INOUT_ARRAY);	
	if (dist_out_arr == NULL)
	{
		Py_DECREF(p1_x_arr);
		Py_DECREF(p1_y_arr);
		Py_DECREF(ind_out_arr);
		PyErr_SetString(PyExc_RuntimeError,"Not able to convert dist_out to NPY_INOUT_ARRAY");
	}
	
	//----- Done checking


	//----- Make c-type easy access arrays
	double *ct_ind = (double *)PyArray_DATA(ind_out_arr);
	double *ct_dist = (double *)PyArray_DATA(dist_out_arr);
	double **ct_p1 = (double **)malloc(2*sizeof(double *));
	ct_p1[0] = (double *)PyArray_DATA(p1_x_arr);
	ct_p1[1] = (double *)PyArray_DATA(p1_y_arr);

	int nP1 = PyArray_SIZE(p1_x_arr);
	int nInd = PyArray_SIZE(ind_out_arr);

	if (nP1 != nInd+1)
	{
		PyErr_SetString(PyExc_RuntimeError,"selfClosest: size of input must be 1 + size of output.");
	}

	int I = 0, J = 0;
	for(I =0; I<nP1-1; I++)
	{
		double minDist = 0;
		int minInd = 0;
		for (J=I+1; J<nP1; J++)
		{
			double dx = (ct_p1[0][I] - ct_p1[0][J]);
			double dy = (ct_p1[1][I] - ct_p1[1][J]);
			// take absolute value
			double adx = dx > 0 ? dx : -dx;
			double ady = dy > 0 ? dy : -dy;
			// Fix for periodicity
			dx = adx > lx/2 ? lx - adx : adx;
			dy = ady > ly/2 ? ly - ady : ady;
			double dist = pow(dx*dx + dy*dy,0.5);
			
			if(J == I+1)	minDist = dist;	// Initialize

			minInd = dist < minDist ? J : minInd;
			minDist= dist < minDist ? dist : minDist;
		}
		ct_ind[I] = minInd;	
		ct_dist[I] = minDist;	
	}


	free(ct_p1);
	Py_DECREF(p1_x_arr);
	Py_DECREF(p1_y_arr);
	Py_DECREF(ind_out_arr);
	Py_DECREF(dist_out_arr);

	/* Build the output tuple */
	int retValue = 0;
	PyObject *ret = Py_BuildValue("i", retValue);
	return ret;
}


static PyObject *cPolyUtils_closest(PyObject *self, PyObject *args)
{
	// all except regions are ndarrays, regions is a list of lists.
	PyObject *p1_x, *p1_y, *p2_x, *p2_y, *ind_out;
	PyObject *p1_x_arr, *p1_y_arr, *p2_x_arr, *p2_y_arr, *ind_out_arr;
	
	// Get list from input arguments
	if (! PyArg_ParseTuple( args, "O!O!O!O!O!", &PyArray_Type, &p1_x, &PyArray_Type, &p1_y, &PyArray_Type, &p2_x, &PyArray_Type, &p2_y, &PyArray_Type, &ind_out )) return NULL;

	//---- Convert, check
	p1_x_arr = PyArray_FROM_OTF(p1_x, NPY_DOUBLE, NPY_IN_ARRAY);	
	if (p1_x_arr == NULL)
	{
		PyErr_SetString(PyExc_RuntimeError,"Not able to convert p1_x to NPY_IN_ARRAY");
	}

	p1_y_arr = PyArray_FROM_OTF(p1_y, NPY_DOUBLE, NPY_IN_ARRAY);	
	if (p1_y_arr == NULL)
	{
		Py_DECREF(p1_x_arr);
		PyErr_SetString(PyExc_RuntimeError,"Not able to convert p1_y to NPY_IN_ARRAY");
	}

	p2_x_arr = PyArray_FROM_OTF(p2_x, NPY_DOUBLE, NPY_IN_ARRAY);	
	if (p2_x_arr == NULL)
	{
		Py_DECREF(p1_x_arr);
		Py_DECREF(p1_y_arr);
		PyErr_SetString(PyExc_RuntimeError,"Not able to convert p2_x to NPY_IN_ARRAY");
	}

	p2_y_arr = PyArray_FROM_OTF(p2_y, NPY_DOUBLE, NPY_IN_ARRAY);	
	if (p2_y_arr == NULL)
	{
		Py_DECREF(p1_x_arr);
		Py_DECREF(p1_y_arr);
		Py_DECREF(p2_x_arr);
		PyErr_SetString(PyExc_RuntimeError,"Not able to convert p2_y to NPY_IN_ARRAY");
	}



	ind_out_arr = PyArray_FROM_OTF(ind_out, NPY_DOUBLE, NPY_INOUT_ARRAY);	
	if (ind_out_arr == NULL)
	{
		Py_DECREF(p1_x_arr);
		Py_DECREF(p1_y_arr);
		Py_DECREF(p2_x_arr);
		Py_DECREF(p2_y_arr);
		PyErr_SetString(PyExc_RuntimeError,"Not able to convert ind_out to NPY_INOUT_ARRAY");
	}
	
	//----- Done checking


	//----- Make c-type easy access arrays
	double *ct_ind = (double *)PyArray_DATA(ind_out_arr);
	double **ct_p1 = (double **)malloc(2*sizeof(double *));
	double **ct_p2 = (double **)malloc(2*sizeof(double *));
	ct_p1[0] = (double *)PyArray_DATA(p1_x_arr);
	ct_p1[1] = (double *)PyArray_DATA(p1_y_arr);
	ct_p2[0] = (double *)PyArray_DATA(p2_x_arr);
	ct_p2[1] = (double *)PyArray_DATA(p2_y_arr);




	int nP1 = PyArray_SIZE(p1_x_arr);
	int nP2 = PyArray_SIZE(p2_x_arr);
	int nInd = PyArray_SIZE(ind_out_arr);

	int iP1 = 0, iP2 = 0;
	for(iP1 =0; iP1<nP1; iP1++)
	{
		double minDist = 0;
		int minInd = 0;
		for (iP2=0; iP2<nP2; iP2++)
		{
			double dist2 = (ct_p1[0][iP1] - ct_p2[0][iP2])*(ct_p1[0][iP1] - ct_p2[0][iP2]);
			dist2 += (ct_p1[1][iP1] - ct_p2[1][iP2])*(ct_p1[1][iP1] - ct_p2[1][iP2]);

			if(iP2 == 0)	minDist = dist2;	// Initialize

			minInd = dist2 < minDist ? iP2 : minInd;
			minDist= dist2 < minDist ? dist2 : minDist;
		}
		ct_ind[iP1] = minInd;	
	}


	free(ct_p1);
	free(ct_p2);
	Py_DECREF(p1_x_arr);
	Py_DECREF(p1_y_arr);
	Py_DECREF(p2_x_arr);
	Py_DECREF(p2_y_arr);
	Py_DECREF(ind_out_arr);

	/* Build the output tuple */
	int retValue = 0;
	PyObject *ret = Py_BuildValue("i", retValue);
	return ret;
}


static PyObject *cPolyUtils_vorCentroid(PyObject *self, PyObject *args)
{
	int start, end;
	// all except regions are ndarrays, regions is a list of lists.
	PyObject *point_region, *regions, *points_x, *points_y, *verts_x, *verts_y, *out_x, *out_y;
	PyObject *point_region_arr, *points_x_arr, *points_y_arr, *verts_x_arr, *verts_y_arr, *out_x_arr, *out_y_arr;

	// Get list from input arguments
	if (! PyArg_ParseTuple( args, "iiO!O!O!O!O!O!O!O!", &start,&end,&PyList_Type, &regions, &PyArray_Type, &point_region, &PyArray_Type, &points_x, &PyArray_Type, &points_y, &PyArray_Type, &verts_x, &PyArray_Type, &verts_y, &PyArray_Type, &out_x, &PyArray_Type, &out_y )) return NULL;

	//---- Convert, check
	point_region_arr = PyArray_FROM_OTF(point_region, NPY_DOUBLE, NPY_IN_ARRAY);	
	if (point_region_arr == NULL)
	{
		PyErr_SetString(PyExc_RuntimeError,"Not able to convert point_region to NPY_IN_ARRAY");
	}
	
	verts_x_arr = PyArray_FROM_OTF(verts_x, NPY_DOUBLE, NPY_IN_ARRAY);	
	if (verts_x_arr == NULL)
	{
		Py_DECREF(point_region_arr);
		PyErr_SetString(PyExc_RuntimeError,"Not able to convert verts_x to NPY_IN_ARRAY");
	}

	verts_y_arr = PyArray_FROM_OTF(verts_y, NPY_DOUBLE, NPY_IN_ARRAY);	
	if (verts_y_arr == NULL)
	{
		Py_DECREF(point_region_arr);
		Py_DECREF(verts_x_arr);
		PyErr_SetString(PyExc_RuntimeError,"Not able to convert verts_y to NPY_IN_ARRAY");
	}

	out_x_arr = PyArray_FROM_OTF(out_x, NPY_DOUBLE, NPY_INOUT_ARRAY);	// Output array
	if (out_x_arr == NULL)
	{
		Py_DECREF(point_region_arr);
		Py_DECREF(verts_x_arr);
		Py_DECREF(verts_y_arr);
		PyErr_SetString(PyExc_RuntimeError,"Not able to convert out_x to NPY_INOUT_ARRAY");
	}

	out_y_arr = PyArray_FROM_OTF(out_y, NPY_DOUBLE, NPY_INOUT_ARRAY);	// Output array
	if (out_y_arr == NULL)
	{
		Py_DECREF(point_region_arr);
		Py_DECREF(verts_x_arr);
		Py_DECREF(verts_y_arr);
		Py_DECREF(out_x_arr);
		PyErr_SetString(PyExc_RuntimeError,"Not able to convert out_y to NPY_INOUT_ARRAY");
	}

	points_x_arr = PyArray_FROM_OTF(points_x, NPY_DOUBLE, NPY_IN_ARRAY);	
	if (points_x_arr == NULL)
	{
		Py_DECREF(point_region_arr);
		Py_DECREF(verts_x_arr);
		Py_DECREF(verts_y_arr);
		Py_DECREF(out_x_arr);
		Py_DECREF(out_y_arr);
		PyErr_SetString(PyExc_RuntimeError,"Not able to convert points_x to NPY_IN_ARRAY");
	}

	points_y_arr = PyArray_FROM_OTF(points_y, NPY_DOUBLE, NPY_IN_ARRAY);	
	if (points_y_arr == NULL)
	{
		Py_DECREF(point_region_arr);
		Py_DECREF(verts_x_arr);
		Py_DECREF(verts_y_arr);
		Py_DECREF(out_x_arr);
		Py_DECREF(out_y_arr);
		Py_DECREF(points_x_arr);
		PyErr_SetString(PyExc_RuntimeError,"Not able to convert points_y to NPY_IN_ARRAY");
	}

	//----- Done checking

	//----- Make c-type easy access arrays
	double *ct_pt_reg = (double *)PyArray_DATA(point_region_arr);
	double *ct_vx = (double *)PyArray_DATA(verts_x_arr);
	double *ct_vy = (double *)PyArray_DATA(verts_y_arr);
	double *ct_px = (double *)PyArray_DATA(points_x_arr);
	double *ct_py = (double *)PyArray_DATA(points_y_arr);
	double *ct_ox = (double *)PyArray_DATA(out_x_arr);
	double *ct_oy = (double *)PyArray_DATA(out_y_arr);

	// Loop over all points in the requested range
	int pt = 0;
	for(pt=start; pt<end; pt++)
	{
		// Get region number corresponding to the point
		int regNum = (int)ct_pt_reg[pt];
		// Get the list of all vertices that make the region
		PyObject * vert_list;
		vert_list = PyList_GetItem(regions,regNum);
		// Loop the list and find all the vertex coordinates (real space)
		// The first coordinate is repeated to close the region.
		int vert_list_len = PyList_Size(vert_list);		// Number of vertices in the list
		double *reg_verts_x = (double *)malloc((vert_list_len+1)*sizeof(double));
		double *reg_verts_y = (double *)malloc((vert_list_len+1)*sizeof(double));
		int isOpen = 0;
		{
			int j = 0;
			PyObject * vert_num;
			for(j=0; j<vert_list_len; j++)
			{
				vert_num = PyList_GetItem(vert_list, j);
				int vert_num_int = PyInt_AsLong(vert_num);
				if (vert_num_int == -1)	// Open region
				{
					isOpen = 1; 
					break;
				}
				// Get the real space coordinates of the vertex
				reg_verts_x[j] = ct_vx[vert_num_int];
				reg_verts_y[j] = ct_vy[vert_num_int];
			}

			// Close the region
			reg_verts_x[vert_list_len] = reg_verts_x[0];
			reg_verts_y[vert_list_len] = reg_verts_y[0];
		}

		// Loop to calculate the centroid. If region is open then return the original point 
		// as the centroid
		double centroid_x =0, centroid_y = 0;
		if(isOpen == 1)
		{
			centroid_x = ct_px[pt];
			centroid_y = ct_py[pt];
		}
		else
		{
			int j = 0;
			double twice_area = 0;
			for(j=0; j<vert_list_len; j++)
			{
				double t = (reg_verts_x[j]*reg_verts_y[j+1] - reg_verts_x[j+1]*reg_verts_y[j]);
				centroid_x += (reg_verts_x[j] + reg_verts_x[j+1])*t;
				centroid_y += (reg_verts_y[j] + reg_verts_y[j+1])*t;
				twice_area += t;
			}
			centroid_x = centroid_x/(3.0*twice_area);
			centroid_y = centroid_y/(3.0*twice_area);
		}

		// Set result
		ct_ox[pt-start] = centroid_x;
		ct_oy[pt-start] = centroid_y;

		free(reg_verts_x);
		free(reg_verts_y);

	}// ROF

	// Clean up
	Py_DECREF(point_region_arr);
	Py_DECREF(verts_x_arr);
	Py_DECREF(verts_y_arr);
	Py_DECREF(out_x_arr);
	Py_DECREF(out_y_arr);
	Py_DECREF(points_x_arr);
	Py_DECREF(points_y_arr);

	/* Build the output tuple */
	int retValue = 0;
	PyObject *ret = Py_BuildValue("i", retValue);
	return ret;
}
