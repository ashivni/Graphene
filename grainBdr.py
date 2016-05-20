# Cite as Ophus, Colin, Ashivni Shekhawat, Haider I. Rasool, and Alex Zettl. "Large-Scale Experimental and Theoretical Study of Graphene Grain Boundary Structures." Phys. Rev. B 92, 205402 (2015).
# arxiv preprint at http://arxiv.org/abs/1508.00497

# This is "version 2" of the script. It has been updated to generate 2D periodic GBs 
# with the function twoPeriodicGB

# Report any bugs/issues to shekhawat.ashivni@gmail.com

import numpy
import scipy
import random
import ase  
from ase import Atoms
from scipy.spatial import Voronoi
import copy
import _cGBUtils
from fractions import Fraction

def writeLammpsData(cr,fName):
	f = open(fName,'w+')

	f.write('%s (written by ASE)\n\n'%(fName))
	f.write('%d\tatoms\n'%(len(cr)))
	f.write('1\tatom types\n')
	f.write('0.0\t%.9g\txlo xhi\n'%(cr.cell[0,0]))
	f.write('0.0\t%.9g\tylo yhi\n'%(cr.cell[1,1]))
	f.write('0.0\t%.9g\tzlo zhi\n'%(cr.cell[2,2]))

	f.write('\n\n')
	f.write('Atoms\n\n')

	for i in range(len(cr)):
		f.write('\t%d\t1 %.9g %.9g %.9g\n'%(i+1,cr.positions[i,0],cr.positions[i,1],cr.positions[i,2]))

	f.close()

def th2V(th):
	# Given angle th (in radians), returns matrix with unit vectors rotated about z by angle th as column vectors
	c , s = numpy.cos(th), numpy.sin(th)
	v1 = numpy.array([c, s,0.0])
	v2 = numpy.array([-s, c,0.0])
	v3 = numpy.array([0,0,1.0])
	V = numpy.zeros((3,3))
	V[:,0], V[:,1], V[:,2] = v1, v2, v3
	return V

def rotatedCrystal(V,size=(2,2,1),a=1.3968418):
	"""
	Generates a triangular crystal lattice of the given size and rotates it so that the new unit vectors 
	align with the columns of V. The positions are set so that the center atom is at the 
	origin. Size is expected to be even in all directions.
	'a' is the atomic distance between the atoms of the hexagonal lattice daul to this crystal.
	In other words, a*sqrt(3) is the lattice constant of the triangular lattice.
	The returned object is of ase.Atoms type
	"""
	numbers = [6.0]
	cell = numpy.array([[a*(3.0**0.5),0,0],[0.5*a*(3.0**0.5),1.5*a,0],[0,0,10*a]])
	positions = numpy.array([[0,0,0]])
	cr = ase.Atoms(numbers=numbers,positions=positions,cell=cell,pbc=[True,True,True])

	# Repeating
	ix = numpy.indices(size, dtype=int).reshape(3,-1)
	tvecs = numpy.einsum('ki,kj',ix,cr.cell)
	rPos = numpy.ndarray((len(cr)*len(tvecs),3))
	for i in range(len(cr)):
		rPos[i*len(tvecs):(i+1)*len(tvecs)] = tvecs + cr.positions[i]
	# New cell size
	for i in range(3):
		cr.cell[i]*=size[i]

	cr = Atoms(symbols=['C']*len(rPos), positions=rPos, cell = cr.cell, pbc=[True,True,True])
	center = numpy.sum(cr.cell,axis=0)*0.5
	cr.positions = cr.positions - center

	cr.cell = numpy.einsum('ik,jk',cr.cell,V)
	cr.positions = numpy.einsum('ik,jk',cr.positions,V)

	return cr

def lowStrainApproximation(l1,l2,ep_max=0.0001,d_start=1,d_fac=2):
	"""
	Find r1, r2 such that abs(0.5*(l1*r1 - l2*r2)/min(l2*r2,l1*r1)) < ep_max
	"""

	ep = 1.0	# strain
	d = d_start
	while abs(ep) > ep_max:
		r1 = Fraction(l2/l1).limit_denominator(d).numerator
		r2 = Fraction(l2/l1).limit_denominator(d).denominator
		if r1 > d:
			r2 = Fraction(l1/l2).limit_denominator(d).numerator
			r1 = Fraction(l1/l2).limit_denominator(d).denominator
		if r1 > d:
			print (l1, l2)
			raise Exception('Cannot find good approximation')
		if r2 == 0 or r1 == 0:
			d *= d_fac
		else:
			ep = 0.5*(l1*r1 - l2*r2)/min(l2*r2,l1*r1)
			if abs(ep) > ep_max:
				d *= d_fac

	return r1, r2

def cvtRelax(gr,**kwargs):
	"""
	Relax a triangular lattice by the cvt algorithm
	"""
	maxIter = kwargs.get('maxIter',10000)
	tol = kwargs.get('tol',1E-5)
	verbose = kwargs.get('verbose',False)

	positions = gr.positions.copy()

	repeat = True
	bulk = positions[gr.get_tags()==0,0:2]		# Bulk points
	pad = positions[gr.get_tags()==2,0:2]		# Pad points
	bdry = positions[gr.get_tags()==1,0:2]		# boundary points

	itNum = 0

	# periodic pad points 
	# we only take periodic image of a strip of width sw (strip width)
	sw = kwargs.get('sw',10.0)
	ix = numpy.array([[0,0],[-1,1]])

	tvecs = numpy.einsum('ki,kj',ix,gr.cell[:2,:2])
	pad_p = numpy.tile(pad.T,len(tvecs)).T + tvecs.repeat(len(pad),axis=0)	# Periodic immutable points, without the central peice
	# cut area outside the strip
	pad_p = pad_p[ numpy.logical_and(* [ numpy.logical_and(pad_p[:,i]>-sw, pad_p[:,i] <= gr.cell[i,i]+sw) for i in range(2) ] )]
	pad_p = numpy.concatenate((pad_p,pad),axis=0)	# adding the central peice

	maxUpd = 1.0	# Magnitude of the maximum update 
	while maxUpd > tol and itNum < maxIter:
		itNum += 1
		if verbose and itNum % 10 == 0:
			print (itNum,maxUpd)
		# Make periodic
		bdry_p = numpy.tile(bdry.T,len(tvecs)).T + tvecs.repeat(len(bdry),axis=0)	# Periodic mutable points, without the central peice
		bdry_p = bdry_p[ numpy.logical_and(* [ numpy.logical_and(bdry_p[:,i]>-sw, bdry_p[:,i] <= gr.cell[i,i]+sw) for i in range(2) ] )]
		bdry_p = numpy.concatenate((bdry_p,bdry),axis=0)	# adding the central peice

		Z = numpy.concatenate((pad_p,bdry_p),axis=0)

		#--- Obtain voronoi tessellation
		vor = Voronoi(Z)

		# Calculate c update
		ox = numpy.zeros(len(bdry))
		oy = numpy.zeros(len(bdry))
		_cGBUtils.vorCentroid(len(Z) - len(bdry), len(Z),vor.regions,vor.point_region,vor.points[:,0],vor.points[:,1],vor.vertices[:,0],vor.vertices[:,1],ox,oy)

		# Calculate the maximum update magnitude
		maxUpd = max( ( (Z[len(Z) - len(bdry):,0] - ox)**2.0 + (Z[len(Z) - len(bdry):,1] - oy)**2.0)**0.5)
		Z[len(Z) - len(bdry):,0] = ox
		Z[len(Z) - len(bdry):,1] = oy


		# get the mutable "non-periodic" positions back
		bdry = Z[len(Z) - len(bdry):,:]

		# get back in the periodic box
		bdry[bdry[:,0]<0,0] += gr.cell[0,0]
		bdry[bdry[:,1]<0,1] += gr.cell[1,1]
		bdry[bdry[:,0]>=gr.cell[0,0],0] -= gr.cell[0,0]
		bdry[bdry[:,1]>=gr.cell[1,1],1] -= gr.cell[1,1]

	if verbose:
		print "Iteration ended with maxUpd = %f, itNum = %d"%(maxUpd,itNum)

	# choose those in box
	#Z = Z[ numpy.logical_and(* [ numpy.logical_and(Z[:,i]>=0, Z[:,i] <= gr.cell[i,i]) for i in range(2) ] )]
	Z = numpy.concatenate((bulk,pad),axis=0)
	Z = numpy.concatenate((Z,bdry),axis=0)
	tags = numpy.zeros(len(Z)).astype('int')
	tags[-len(pad)-len(bdry):] = 2
	tags[-len(bdry):] = 1

	pos = numpy.concatenate((Z,numpy.zeros(len(Z)).reshape((len(Z),1))),1)
	cr = ase.Atoms(numbers=[6]*len(pos),positions=pos,tags=tags,cell=gr.cell,pbc=[True,True,True])
	
	return cr

def centroid2Vertices(vor,L):
	"""
	Given a set of centroids (voronoi generators) return the vertices.
	A cell of size L[0], L[1] is assumed. data is assumed to be 
	"""
	# vertices
	v = vor.vertices

	# choose those in box
	v = v[ numpy.logical_and(* [ numpy.logical_and(v[:,i]>=0, v[:,i] <= L[i]) for i in range(2) ] )]

	return v

def anamVertexRegionSize(vor, L, typical=6):
	"""
	Given a set of centroids (voronoi generators),
	for each vertex in the tessellation, return 1 if the vertex is part of a region that 
	is not of the 'typical' size
	"""
	# vertices
	v = vor.vertices

	sizeFlag = numpy.zeros(len(v))		# flag for anamolous region size
	#regSize = [[]]*len(v)		# list of region sizes
	for reg in vor.regions:
		if len(reg) > 0 and -1 not in reg:	# Non-empty and non-open region
			size = len(reg)
			if size != typical:
				for vert in reg:		# update size of all vertices that are in the region
					#regSize[vert].append(size)
					sizeFlag[vert] = 1 

	# Choose all in box
	sizeFlag = sizeFlag[ numpy.logical_and(* [ numpy.logical_and(v[:,i]>=0, v[:,i] <= L[i]) for i in range(2) ] )]

	sizeFlag = sizeFlag.astype('int')
	return sizeFlag

def voronoiCutoff(vor, L, cutoff = 0.1):
	"""
	Given a voronoi object tag the set of vertices that are closer to each other than cutoff. Only
	tag one vertex in each pair.
	"""
	# vertices
	v = vor.vertices

	closeFlag = numpy.zeros(len(v))		# flag for close vertices

	for reg in vor.regions:
		if len(reg) > 0 and -1 not in reg:	# Non-empty and non-open region
			# find the distance between all vertices, accounting for periodicity
			r1 = numpy.array(reg)	 
			r2 = numpy.roll(r1,1)
			d = numpy.sum((vor.vertices[r2] - vor.vertices[r1])**2.0,axis=1)**0.5
			# Choose those vertex pairs that are less than cutoff apart
			r1 = r1[d < cutoff]
			r2 = r2[d < cutoff]
			# From the pair, choose the vertex that has the larger x coordinate
			# If x coordinates are same, choose the larger y coordinate
			if len(r1) > 0:
				flag = numpy.zeros(len(r1))
				for i, r1x, r1y, r2x, r2y in zip(range(len(r1)), vor.vertices[r1][:,0], vor.vertices[r1][:,1], vor.vertices[r2][:,0], vor.vertices[r2][:,1]):
					if r1x > r2x:
						flag[i] = r1[i]
					elif r2x > r1x:
						flag[i] = r2[i]
					elif r1y > r2y:
						flag[i] = r1[i]
					else:
						flag[i] = r2[i]

				flag = flag.astype('int')

				closeFlag[flag] = 1

	# Choose all in box
	closeFlag = closeFlag[ numpy.logical_and(* [ numpy.logical_and(v[:,i]>=0, v[:,i] <= L[i]) for i in range(2) ] )]

	closeFlag = closeFlag.astype('int')
	return closeFlag

def tr2gr(tr,periodic=True,num=[6,6,6],cutoff=0.1,vorMet = True):
	"""
	Given a triagular lattice crystal, returns the corresponding hexagonal (graphene) lattice crystal
	"""
	Z = tr.positions[:,0:2]
	L = tr.cell.diagonal()[:2]
	lx, ly = L
	# Voronoi construction
	if periodic:
		ix = numpy.indices((3,3), dtype=int).reshape(2,-1) -1
		tvecs = numpy.einsum('ki,kj',ix,numpy.diag(L))
		cents = numpy.tile(tvecs.T,len(Z)).T + Z.repeat(9,axis=0)	# centers
	else:
		ix = numpy.array([[0,0,0],[-1,0,1]])
		tvecs = numpy.einsum('ki,kj',ix,numpy.diag(L))
		cents = numpy.tile(tvecs.T,len(Z)).T + Z.repeat(3,axis=0)	# centers
	
	vor = Voronoi(cents)

	v = centroid2Vertices(vor,L)

	tags = anamVertexRegionSize(vor,L)
	pos = numpy.concatenate((v,numpy.zeros(len(v)).reshape((len(v),1))),1)		# 

	# Clear the interior with voronoi
	closeFlag = voronoiCutoff(vor,L,cutoff)

	# Clear the boundaries with c
	# Find the vertices in a thin strip near the boundary of the box
	boundary = numpy.zeros(len(pos))
	px, py = pos[:,0], pos[:,1]
	ds = 1.5*cutoff
	# Left boundary
	boundary[numpy.logical_and(numpy.logical_and(px<ds, px > -ds), numpy.logical_and(py>-ds, py < ly +ds) )] = 1
	# Right boundary
	boundary[numpy.logical_and(numpy.logical_and(px<lx+ds, px > lx-ds), numpy.logical_and(py>-ds, py < ly +ds) )] = 1
	# Bottom boundary
	boundary[numpy.logical_and(numpy.logical_and(py<ds, py > -ds), numpy.logical_and(px>-ds, px < lx +ds) )] = 1
	# Top boundary
	boundary[numpy.logical_and(numpy.logical_and(py<ly+ds, py > ly-ds), numpy.logical_and(px>-ds, px < lx +ds) )] = 1

	#choose positions in the boundary
	bPos = pos[boundary==1,:]
	if len(bPos) > 1:
		# Find minDist with c routine
		minInd = numpy.zeros(len(bPos)-1)
		minDist = numpy.zeros(len(bPos)-1)
		_cGBUtils.selfClosest(tr.cell[0,0],tr.cell[1,1],bPos[:,0],bPos[:,1],minInd,minDist)




		dist = numpy.ones(len(pos))*2*cutoff
		temp = numpy.array(numpy.where(boundary==1)[0]).astype('int')
		dist[temp[:-1]] = minDist
		closeFlag[dist<cutoff] = 1

		tags[boundary==1] = 2

	pos = numpy.delete(pos,numpy.where(closeFlag == 1),axis=0)
	tags = numpy.delete(tags,numpy.where(closeFlag == 1),axis=0)

	numbers = [num[x] for x in tags]
	pos[:,2] = tr.cell[2,2]*0.5
	cr = ase.Atoms(numbers=numbers,positions=pos,cell=tr.cell,tags=tags,pbc=[True,True,True])

	return cr


def twoPeriodicGB(a=1.3968418,N1=numpy.array([1,2]),N2=numpy.array([1,3]),dy = numpy.array([0,0]),cell_width=120.0,ep_max=0.0001,**kwargs):
	"""
	Returns a doubly periodic grain boundary with the two grains oriented such that 
	th1 = (2*N1[0] + N1[1])/(rt3*N1[1])
	th2 = (2*N2[0] + N2[1])/(rt3*N2[1])
	misorientation = th1 + th2
	line_angle = th1 - th2

	The length of the grain boundary along the GB direction is determined as described in the references (see reference at the beginning of this file).
	The length is chosen such that the net strain for non-commensurate boundaries is less that ep_max

	'a' is the carbon-carbon bond length for graphene (Defaults to the bond length predicted by REBO potential)
	"""
	verbose = kwargs.get('verbose',True)
	if cell_width < 120:
		if verbose:
			print "cell_width is small. Consider setting it to be >= 120 Angstroms."
			print "Using small cell_width may give unpredictable results."
			print "Use verbose = False to stop printing this message"
		pad = 5.0
		bWidth = 10.0
		width = cell_width
	else:
		pad = 10.0
		bWidth = 20.0
		width = cell_width 

	ratPad = 2.0
	cutoff = kwargs.get('cutoff',1.0)
	overlap = kwargs.get('overlap',0.5)
	
	# Calculate the repeat distances and angles of the two crystals
	rt3 = 3.0**0.5
	th1 = numpy.arctan((2*N1[0] + N1[1])/(rt3*N1[1]))
	d1 = (N1[0]*N1[0] + N1[1]*N1[1] + N1[0]*N1[1])**0.5
	ly1 = a*rt3*d1
	lx1 = width/2.0

	th2 = numpy.arctan((2*N2[0] + N2[1])/(rt3*N2[1]))
	d2 = (N2[0]*N2[0] + N2[1]*N2[1] + N2[0]*N2[1])**0.5
	ly2 = a*rt3*d2
	lx2 = width/2.0

	# Dimensions of the box
	Lx = width

	# find a good rational approximation such that r1*ly1 =(approx) r2*ly2
	r1, r2 = lowStrainApproximation(ly1,ly2,ep_max=ep_max)

	Ly = 0.5*(ly1*r1 + ly2*r2)

	# Get the number of repeats needed for the crystals
	D = (Lx*Lx + Ly*Ly)**0.5			# Diagonal of the periodic box
	r = (int(D/a) + 2)*3	# repeats
	rc = rotatedCrystal(numpy.eye(3),size=(r,r,1),a=a)
	rc.positions[:,2] = 0
	orgPos = rc.positions

	# centers of the grains
	c1 = numpy.array([lx1/2,Ly/2+dy[0],0])
	c2 = numpy.array([lx1/2 + Lx/2,Ly/2+dy[1],0])

	# Rotated axis vectors
	V1 = th2V(th1)
	V2 = th2V(th2)

	# Grain 1
	p1 = copy.copy(orgPos)
	p1 = numpy.dot(V1,p1.T).T
	p1 += c1		# Center the crystal at the center of the grain
	# Choose all in the box
	p1 = p1[numpy.logical_and(p1[:,0] >=0-overlap, p1[:,0] < lx1+overlap)]
	p1 = p1[numpy.logical_and(p1[:,1] >=0, p1[:,1] <= ly1*r1+1E-7)]
	p1[:,1] *= Ly/(ly1*r1)

	# Grain 2
	p2 = copy.copy(orgPos)
	p2 = numpy.dot(V2,p2.T).T
	p2 += c2		# Center the crystal at the center of the grain
	# Choose all in the box
	p2 = p2[numpy.logical_and(p2[:,0] >=lx1, p2[:,0] < Lx)]
	p2 = p2[numpy.logical_and(p2[:,1] >=0, p2[:,1] <= ly2*r2+1E-7)]
	#strain grain 2
	p2[:,1] *= Ly/(ly2*r2)

	# collect all positions
	p = numpy.concatenate((p1,p2))

	# Shift positions to make the two GBs equidistant from the center
	p[:,0] += lx2/2.0
	# Some atoms spill out of the box bring them back in
	spilled = p[:,0] >= Lx
	p[spilled,0] -= Lx

	# Set tags
	t = numpy.zeros(len(p)).reshape((len(p),1))		# Default tag bulk
	# Grain boundary tags
	t[numpy.logical_and(p[:,0] > lx2/2.0 - bWidth, p[:,0] < lx2/2.0 + bWidth)] = 1
	t[numpy.logical_and(p[:,0] > lx2/2.0 + lx1 - bWidth, p[:,0] < lx2/2.0 + lx1 + bWidth)] = 1

	# Pad tags
	t[numpy.logical_and(p[:,0] <= lx2/2.0 - bWidth, p[:,0] > lx2/2.0 - bWidth - pad)] = 2
	t[numpy.logical_and(p[:,0] >= lx2/2.0 + bWidth, p[:,0] < lx2/2.0 + bWidth + pad)] = 2
	t[numpy.logical_and(p[:,0] <= lx2/2.0 + lx1 - bWidth, p[:,0] > lx2/2.0 + lx1 - bWidth - pad)] = 2
	t[numpy.logical_and(p[:,0] >= lx2/2.0 + lx1 + bWidth, p[:,0] < lx2/2.0 + lx1 + bWidth + pad)] = 2


	# seperate the bulk, pad and gb positions; filter gb for cutoff
	bulkP = p[(t==0)[:,0],:]
	gbP 	= p[(t==1)[:,0],:]
	padP 	= p[(t==2)[:,0],:]

	# Find minDist with c routine
	minInd = numpy.zeros(len(gbP)-1)
	minDist = numpy.zeros(len(gbP)-1)
	_cGBUtils.selfClosest(Lx,Ly,gbP[:,0],gbP[:,1],minInd,minDist)
	gbP = numpy.delete(gbP,numpy.where(minDist < a*cutoff),axis=0)

	# collect all positions
	positions = numpy.concatenate((bulkP,gbP))
	positions = numpy.concatenate((positions,padP))
	tags = numpy.zeros(len(positions))
	tags[-len(gbP)-len(padP):] = 1
	tags[-len(padP):] = 2


	# Find the vertices in a thin strip near the boundary of the box
	boundary = numpy.zeros(len(positions))
	px, py = positions[:,0], positions[:,1]
	ds = 0.15
	# Left boundary
	boundary[numpy.logical_and(numpy.logical_and(px<ds, px > -ds), numpy.logical_and(py>-ds, py < Ly +ds) )] = 1
	# Right boundary
	boundary[numpy.logical_and(numpy.logical_and(px<Lx+ds, px > Lx-ds), numpy.logical_and(py>-ds, py < Ly +ds) )] = 1
	# Bottom boundary
	boundary[numpy.logical_and(numpy.logical_and(py<ds, py > -ds), numpy.logical_and(px>-ds, px < Lx +ds) )] = 1
	# Top boundary
	boundary[numpy.logical_and(numpy.logical_and(py<Ly+ds, py > Ly-ds), numpy.logical_and(px>-ds, px < Lx +ds) )] = 1

	#choose positions in the boundary
	bPos = positions[boundary==1,:]
	if len(bPos) > 1:
		# Find minDist with c routine
		closeFlag = numpy.zeros(len(positions))
		minInd = numpy.zeros(len(bPos)-1)
		minDist = numpy.zeros(len(bPos)-1)
		_cGBUtils.selfClosest(Lx,Ly,bPos[:,0],bPos[:,1],minInd,minDist)


		dist = numpy.ones(len(positions))*2*cutoff
		temp = numpy.array(numpy.where(boundary==1)[0]).astype('int')
		dist[temp[:-1]] = minDist
		closeFlag[dist<cutoff] = 1

		positions = numpy.delete(positions,numpy.where(closeFlag == 1),axis=0)
		tags = numpy.delete(tags,numpy.where(closeFlag == 1),axis=0)


	cell = numpy.diag([Lx,Ly,10])
	numbers = numpy.ones(len(positions))*6
	cr = Atoms(numbers = numbers, positions=positions,tags=tags, cell = cell, pbc=[True,True,True])

	# `Rattlable' atoms
	rattlable = numpy.zeros(len(positions)).reshape((len(positions),1))		# Atoms not rattlable by default
	rattlable[numpy.logical_and(positions[:,0] > lx2/2.0 - (bWidth-ratPad), positions[:,0] < lx2/2.0 + (bWidth-ratPad))] = 1
	rattlable[numpy.logical_and(positions[:,0] > lx1+lx2/2.0 - (bWidth-ratPad), positions[:,0] < lx1+lx2/2.0 + (bWidth-ratPad))] = 1
	cr.rattlable = rattlable.reshape(len(rattlable))

	cr.th1 = th1*180/numpy.pi
	cr.th2 = th2*180/numpy.pi

	tol = kwargs.get('tol',1E-5)
	r_cr = cvtRelax(cr,verbose=verbose,tol=tol)	# Relax GB using CVT algorithm
	cr_final = tr2gr(r_cr,periodic=True)	# Convert to graphene crystal
	cr_final.positions[:,1] *= -1
	cr_final.center()

	return cr_final

def onePeriodicGB(a=1.3968418,N1=numpy.array([1,2]),N2=numpy.array([1,3]),dy = numpy.array([0,0]),cell_width=60.0,ep_max=0.0001,**kwargs):
	"""
	Returns a grain boundary with the two grains oriented such that 
	th1 = (2*N1[0] + N1[1])/(rt3*N1[1])
	th2 = (2*N2[0] + N2[1])/(rt3*N2[1])
	misorientation = th1 + th2
	line_angle = th1 - th2

	The length of the grain boundary along the GB direction is determined as described in the references (see reference at the beginning of this file).
	The length is chosen such that the net strain for non-commensurate boundaries is less that ep_max

	'a' is the carbon-carbon bond length for graphene (Defaults to the bond length predicted by REBO potential)
	"""
	verbose = kwargs.get('verbose',True)
	if cell_width < 60:
		if verbose:
			print "cell_width is small. Consider setting it to be >= 60 Angstroms."
			print "Using small cell_width may give unpredictable results."
			print "Use verbose = False to stop printing this message"
		pad = 5.0
		bWidth = 10.0
		width = 80.0
	else:
		pad = 10.0
		bWidth = 20.0
		width = cell_width + 20.0

	ratPad = 2.0
	cutoff = kwargs.get('cutoff',1.0)
	overlap = kwargs.get('overlap',0.5)
	
	# Calculate the repeat distances and angles of the two crystals
	rt3 = 3.0**0.5
	th1 = numpy.arctan((2*N1[0] + N1[1])/(rt3*N1[1]))
	d1 = (N1[0]*N1[0] + N1[1]*N1[1] + N1[0]*N1[1])**0.5
	ly1 = a*rt3*d1
	lx1 = width/2.0

	th2 = numpy.arctan((2*N2[0] + N2[1])/(rt3*N2[1]))
	d2 = (N2[0]*N2[0] + N2[1]*N2[1] + N2[0]*N2[1])**0.5
	ly2 = a*rt3*d2
	lx2 = width/2.0

	# Dimensions of the box
	Lx = width

	# find a good rational approximation such that r1*ly1 =(approx) r2*ly2
	r1, r2 = lowStrainApproximation(ly1,ly2,ep_max=ep_max)

	Ly = 0.5*(ly1*r1 + ly2*r2)

	# Get the number of repeats needed for the crystals
	D = (Lx*Lx + Ly*Ly)**0.5			# Diagonal of the periodic box
	r = (int(D/a) + 2)*3	# repeats
	rc = rotatedCrystal(numpy.eye(3),size=(r,r,1),a=a)
	rc.positions[:,2] = 0
	orgPos = rc.positions

	# centers of the grains
	c1 = numpy.array([lx1/2,Ly/2+dy[0],0])
	c2 = numpy.array([lx1/2 + Lx/2,Ly/2+dy[1],0])

	# Rotated axis vectors
	V1 = th2V(th1)
	V2 = th2V(th2)

	# Grain 1
	p1 = copy.copy(orgPos)
	p1 = numpy.dot(V1,p1.T).T
	p1 += c1		# Center the crystal at the center of the grain
	# Choose all in the box
	p1 = p1[numpy.logical_and(p1[:,0] >=0, p1[:,0] < lx1+overlap)]
	p1 = p1[numpy.logical_and(p1[:,1] >=0, p1[:,1] <= ly1*r1+1E-7)]
	p1[:,1] *= Ly/(ly1*r1)

	# Grain 2
	p2 = copy.copy(orgPos)
	p2 = numpy.dot(V2,p2.T).T
	p2 += c2		# Center the crystal at the center of the grain
	# Choose all in the box
	p2 = p2[numpy.logical_and(p2[:,0] >=lx1-overlap, p2[:,0] < Lx)]
	p2 = p2[numpy.logical_and(p2[:,1] >=0, p2[:,1] <= ly2*r2+1E-7)]
	#strain grain 2
	p2[:,1] *= Ly/(ly2*r2)

	# collect all positions
	p = numpy.concatenate((p1,p2))

	# Set tags
	t = numpy.zeros(len(p)).reshape((len(p),1))		# Default tag bulk
	# Grain boundary tags
	t[numpy.logical_and(p[:,0] > lx1 - bWidth, p[:,0] < lx1 + bWidth)] = 1

	# Pad tags
	t[numpy.logical_and(p[:,0] <= lx1 - bWidth, p[:,0] > lx1 - bWidth - pad)] = 2
	t[numpy.logical_and(p[:,0] >= lx1 + bWidth, p[:,0] < lx1 + bWidth + pad)] = 2


	# seperate the bulk, pad and gb positions; filter gb for cutoff
	bulkP = p[(t==0)[:,0],:]
	gbP 	= p[(t==1)[:,0],:]
	padP 	= p[(t==2)[:,0],:]

	# Find minDist with c routine
	minInd = numpy.zeros(len(gbP)-1)
	minDist = numpy.zeros(len(gbP)-1)
	_cGBUtils.selfClosest(Lx,Ly,gbP[:,0],gbP[:,1],minInd,minDist)
	gbP = numpy.delete(gbP,numpy.where(minDist < a*cutoff),axis=0)

	# collect all positions
	positions = numpy.concatenate((bulkP,gbP))
	positions = numpy.concatenate((positions,padP))
	tags = numpy.zeros(len(positions))
	tags[-len(gbP)-len(padP):] = 1
	tags[-len(padP):] = 2


	# Find the vertices in a thin strip near the boundary of the box
	boundary = numpy.zeros(len(positions))
	px, py = positions[:,0], positions[:,1]
	ds = 0.15
	# Left boundary
	boundary[numpy.logical_and(numpy.logical_and(px<ds, px > -ds), numpy.logical_and(py>-ds, py < Ly +ds) )] = 1
	# Right boundary
	boundary[numpy.logical_and(numpy.logical_and(px<Lx+ds, px > Lx-ds), numpy.logical_and(py>-ds, py < Ly +ds) )] = 1
	# Bottom boundary
	boundary[numpy.logical_and(numpy.logical_and(py<ds, py > -ds), numpy.logical_and(px>-ds, px < Lx +ds) )] = 1
	# Top boundary
	boundary[numpy.logical_and(numpy.logical_and(py<Ly+ds, py > Ly-ds), numpy.logical_and(px>-ds, px < Lx +ds) )] = 1

	#choose positions in the boundary
	bPos = positions[boundary==1,:]
	if len(bPos) > 1:
		# Find minDist with c routine
		closeFlag = numpy.zeros(len(positions))
		minInd = numpy.zeros(len(bPos)-1)
		minDist = numpy.zeros(len(bPos)-1)
		_cGBUtils.selfClosest(Lx,Ly,bPos[:,0],bPos[:,1],minInd,minDist)


		dist = numpy.ones(len(positions))*2*cutoff
		temp = numpy.array(numpy.where(boundary==1)[0]).astype('int')
		dist[temp[:-1]] = minDist
		closeFlag[dist<cutoff] = 1

		positions = numpy.delete(positions,numpy.where(closeFlag == 1),axis=0)
		tags = numpy.delete(tags,numpy.where(closeFlag == 1),axis=0)


	cell = numpy.diag([Lx,Ly,10])
	numbers = numpy.ones(len(positions))*6
	cr = Atoms(numbers = numbers, positions=positions,tags=tags, cell = cell, pbc=[True,True,True])

	# `Rattlable' atoms
	rattlable = numpy.zeros(len(positions)).reshape((len(positions),1))		# Atoms not rattlable by default
	#rattlable[numpy.logical_or(positions[:,0] < bWidth-ratPad, positions[:,0] > Lx - (bWidth-ratPad))] = 1
	rattlable[numpy.logical_and(positions[:,0] > lx1 - (bWidth-ratPad), positions[:,0] < lx1 + (bWidth-ratPad))] = 1
	cr.rattlable = rattlable.reshape(len(rattlable))

	cr.th1 = th1*180/numpy.pi
	cr.th2 = th2*180/numpy.pi

	r_cr = cvtRelax(cr,verbose=verbose)	# Relax GB using CVT algorithm
	gr_cr =tr2gr(r_cr,periodic=False)	# Convert to graphene crystal
	
	# Delete atoms to get the correct GB width

	pos = gr_cr.positions
	tags = gr_cr.get_tags()
	mask = numpy.logical_or(pos[:,0] > gr_cr.cell[0,0]*0.5 + cell_width*0.5, pos[:,0] < gr_cr.cell[0,0]*0.5 - cell_width*0.5)
	pos = numpy.delete(pos,numpy.where(mask),axis=0)
	tags = numpy.delete(tags,numpy.where(mask),axis=0)
	numbers = numpy.ones(len(pos))*6
	cell = numpy.diag([cell_width,Ly,50])
	cr_final = Atoms(numbers = numbers, positions=pos,tags=tags, cell = cell, pbc=[True,True,True])
	# Flip to correct the angles
	cr_final.positions[:,1] *= -1
	cr_final.center()

	return cr_final

