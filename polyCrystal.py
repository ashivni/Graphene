# Cite as Ashivni Shekhawat, Robert O. Ritchie, "Toughenss and Strength of Nanocrystalline Graphene", Nature Communications

# Report any bugs/issues to shekhawat.ashivni@gmail.com
import numpy
import scipy
import random
import fractions
import ase  
from fractions import Fraction
from ase import Atoms
from scipy.spatial import Voronoi
import copy
import _cPolyUtils
import warnings

def periodicCrystal(L=numpy.array([100,100]),N=4):
	tr = periodicVoronoiCell(L=L,N=N)
	tr_rel = cvtRelax(tr,verbose=False,tol=1e-5)
	cr = tr2gr(tr_rel)

	return cr

def rotatedCrystal(V,size=(2,2,1),a=1.3968418,cType='gr'):
	"""
	Generates a triangular crystal lattice of the given size and rotates it so that the new unit vectors 
	align with the columns of V. The positions are set so that the center atom is at the 
	origin. Size is expected to be even in all directions.
	'a' is the atomic distance between the atoms of the hexagonal lattice daul to this crystal.
	In other words, a*sqrt(3) is the lattice constant of the triangular lattice.
	The returned object is of ase.Atoms type
	"""
	if cType == 'gr':
		cr = GB.grapheneCrystal(1,1,'armChair').aseCrystal(ccBond=a)
	else:
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

def ptsInBox(N,L=numpy.ones(2)):
	# Returns N uniformly distributed points in the orthogonal box with sizes L[i]
	return [numpy.random.uniform(size=N)*L[i] for i in range(2)]

def randomAxes(N):
	# Returns N randomly oriented axis. The unit vectors are the columns
	ax = []
	for i in range(N):
		v1 = numpy.random.normal(size=3)		# Choose a random point on the unit circle; this is the x-axis
		v1[2] = 0
		v1 = v1/(sum(v1**2.0)**0.5)
		v2 = numpy.array([-v1[1],v1[0],0]) 		# This is the second unit vector
		v3 = numpy.array([0,0,1])
		V = numpy.zeros((3,3))
		V[:,0], V[:,1], V[:,2] = v1, v2, v3
		ax.append(V)

	return ax

def vorEdges(vor, far):
	"""
	Given a voronoi tesselation, retuns the set of voronoi edges.
	far is the length of the "infinity" edges
	"""
	edges = []
	for simplex in vor.ridge_vertices:
		simplex = numpy.asarray(simplex)
		if numpy.all(simplex >= 0):
			edge = {}
			edge['p1'], edge['p2'] = vor.vertices[simplex,0], vor.vertices[simplex,1]
			edge['p1'] = numpy.array([vor.vertices[simplex,0][0],vor.vertices[simplex,1][0]])
			edge['p2'] = numpy.array([vor.vertices[simplex,0][1],vor.vertices[simplex,1][1]])
			edge['t'] = (edge['p2'] - edge['p1'])/numpy.linalg.norm(edge['p2'] - edge['p1'])
			edges.append(edge)


	ptp_bound = vor.points.ptp(axis=0)
	center = vor.points.mean(axis=0)
	for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
		simplex = numpy.asarray(simplex)
		if numpy.any(simplex < 0):
			i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

			t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
			t /= numpy.linalg.norm(t)
			n = numpy.array([-t[1], t[0]])  # normal

			midpoint = vor.points[pointidx].mean(axis=0)
			direction = numpy.sign(numpy.dot(midpoint - center, n)) * n

			far_point = vor.vertices[i] + direction * ptp_bound.max()*far
			edge = {}
			edge['p1'], edge['p2'] = numpy.array([vor.vertices[i,0], far_point[0]]),numpy.array([vor.vertices[i,1], far_point[1]])
			edge['p1'], edge['p2'] = vor.vertices[i,:],far_point
			edge['t'] = (edge['p2'] - edge['p1'])/numpy.linalg.norm(edge['p2'] - edge['p1'])
			edges.append(edge)
	return edges
	
def pointsInRegion(regNum,vor,p,overlap = 0.0):
	"""
	returns the subset of points p that are inside the regNum region of the voronoi object 
	vor. The boundaries of the region are extended by an amount given by 'overlap'. 
	"""
	reg = vor.regions[vor.point_region[regNum]]	# region associated with the point
	if -1 in reg:
		raise Exception('Open region associated with generator')
	nVerts = len(reg)	# number of verticies in the region
	p0 = vor.points[regNum]

	for i in range(len(reg)):
		vert1, vert2 = vor.vertices[reg[i]], vor.vertices[reg[(i+1)%len(reg)]]
		dr = vert1 - vert2	# edge
		dr = dr/numpy.linalg.norm(dr)	# normalize
		dn = numpy.array([dr[1],-dr[0]])	# normal to edge
		dn = dn if numpy.dot(dn,vert2-p0[:2]) > 0 else -dn	# orient so that the normal is outwards
		d1 = numpy.einsum('i,ji',dn,vert2+dn*overlap - p[:,:2])
		p = p[ d1*numpy.dot(dn,vert2 - p0[:2]) > 0 ]

	return p

def periodicVoronoiCell(a=1.3968418,L=numpy.array([10,10]),N=10,cType='tr',**kwargs):
	"""
	Returns a cell of size L with N grains made according to the voronoi constructions.
	The positions and orientations of the grains are random.
	"""
	# Get a list of axes
	ax = kwargs.get('axes',randomAxes(N))
	# Get origins
	if 'x0' not in kwargs:
		x0, y0 = ptsInBox(N,L)
	else:
		x0 = kwargs['x0']
		y0 = kwargs['y0']
		N = len(x0)
	z0 = numpy.zeros(N)
	# Repeat for double periodic geometry
	ax = ax*9	# Repeat 9 times
	x, y, z = [], [], [] 
	for i in [0,-1,1]:
		for j in [0,-1,1]:
			x.extend(x0+L[0]*i)
			y.extend(y0+L[1]*j)
			z.extend(z0)
	
	x, y, z = numpy.array(x),  numpy.array(y), numpy.array(z)
	orgs = numpy.concatenate((x.reshape(N*9,1),y.reshape(N*9,1),z.reshape(N*9,1)),axis=1)

	# Get the number of repeats needed for the crystals
	D = numpy.linalg.norm(L)*3			# Diagonal of the periodic box
	r = (int(D/a) + 2)*2	# repeats
	rc = rotatedCrystal(numpy.eye(3),size=(r,r,1),a=a,cType=cType)
	rc.positions[:,2] = 0
	orgPos = rc.positions

	#---- Find the grain boundaries with a voronoi tesselation
	#---- we do this to mark the atoms near the grain boundary.
	centers = numpy.zeros((len(x),2))
	centers[:,0], centers[:,1] = x, y
	vor = Voronoi(centers)
	edges = vorEdges(vor,D)


	#positions = []	# positions of the atoms in the cell
	bulkPositions = numpy.ndarray(shape=(0,3))	# positions of the bulk atoms in the cell
	gbPositions = numpy.ndarray(shape=(0,3))		# positions of the grain boundary atoms in the cell
	unRatPositions = numpy.ndarray(shape=(0,3))	# positions of the unrattlable atoms in the cell
	padPositions = numpy.ndarray(shape=(0,3))		# positions of the pad atoms in the cell

	tags = numpy.ndarray(shape=(0,1))	# tags of the atoms 0 = bulk, 1 = grain boundary, 2 = pad region, 3 = unrattlable
	gn = 0
	overlap = kwargs.get('overlap',0.5)
	for v, x1, y1, z1 in zip(ax,x0,y0,z0):
		gn += 1
		p0 = numpy.array([x1,y1,z1])
		p = copy.copy(orgPos)
		p = numpy.dot(v,p.T).T
		p += p0		# Center the crystal at the center of the grain
		# Choose all in the big box
		p = p[ numpy.logical_and(* [ numpy.logical_and(p[:,i]>-L[i], p[:,i] <= 2*L[i]) for i in range(2) ] )]

		# Find the grain number to which we are closest
		#allDist = cdist(p, orgs, 'euclidean')
		#gNum = numpy.argmin(allDist,axis=1)

		if overlap == 0:
			# Find gNum with c routine
			gNum = numpy.zeros(len(p))
			_cPolyUtils.closest(p[:,0],p[:,1],orgs[:,0],orgs[:,1],gNum)
			gNum = gNum.astype(int)

			p = p[gNum == gn-1]	# Choose all that belong to the grain
		else:
			# finite overlap means we need to use voronoi edges to determine which points lie within an overlap region
			# voronoi region associated with the point
			reg = vor.regions[vor.point_region[gn-1]]	# region associated with the point
			if -1 in reg:
				raise Exception('Open region associated with generator')
			nVerts = len(reg)	# number of verticies in the region

			p = pointsInRegion(gn-1,vor,p,overlap=overlap)

		# Correct for periodic boundaries
		for i in range(2):
			p[p[:,i] < 0,i] += L[i]
			p[p[:,i] >= L[i],i] -= L[i]
		# Find a strip close to the grain boundary
		t = numpy.zeros(len(p)).reshape((len(p),1))		# Default tag bulk
		bWidth = kwargs.get('bWidth',10.0)
		pad = kwargs.get('pad',5.0)
		ratPad = kwargs.get('ratPad',2.0)
		for edge in edges:
		#for edge in pt2edge[(x1,y1,z1)]:
			p1, p2, slope = [numpy.ndarray((1,3))*0 for i in range(3)]
			p1[:,:2], p2[:,:2], slope[:,:2] = edge['p1'], edge['p2'],edge['t']

			# The library sometimes returns nan due to the 2d nature of the problem. Reset it. 
			slope[:,2] = 0
			p1[:,2] = 0
			p2[:,2] = 0

			proj1, proj2 = numpy.dot(p-p1,slope.T), numpy.dot(p-p2,slope.T)
			distToEdge = (numpy.sum( ((p1 - p) + proj1*slope)**2.0,axis=1)**0.5).reshape((len(p),1))
			# tag pad region
			t [numpy.logical_and(numpy.logical_and(t!=1,t!=3), numpy.logical_and(numpy.logical_and( distToEdge<bWidth+pad, distToEdge >= bWidth), proj1*proj2 < 0 ))] = 2
			# tag unrattlable
			t [numpy.logical_and(t!=1,numpy.logical_and(numpy.logical_and(distToEdge<bWidth, distToEdge > bWidth-ratPad), proj1*proj2 < 0))] = 3
			# tag grain boundary
			t [numpy.logical_and(distToEdge<=bWidth-ratPad, proj1*proj2 < 0 )] = 1
		bulkPositions = numpy.concatenate((bulkPositions,p[(t==0)[:,0],:]))
		padPositions = numpy.concatenate((padPositions,p[(t==2)[:,0],:]))
		unRatPositions = numpy.concatenate((unRatPositions,p[(t==3)[:,0],:]))

		# Filter out 'close' atoms in grain boundary
		cutoff = kwargs.get('cutoff',0.2)
		thisGB = p[(t==1)[:,0],:]
		if len(gbPositions) > 0:
			#gbDist = cdist(thisGB,gbPositions)
			#minDist = numpy.min(gbDist,axis=1)

			# Find minDist with c routine
			minInd = numpy.zeros(len(thisGB)).astype('int')
			_cPolyUtils.closest(thisGB[:,0],thisGB[:,1],gbPositions[:,0],gbPositions[:,1],minInd)
			minDist = numpy.sum((thisGB - gbPositions[minInd])**2.0,axis=1)**0.5

			thisGB = numpy.delete(thisGB,numpy.where(minDist < a*cutoff),axis=0)
		gbPositions = numpy.concatenate((gbPositions,thisGB))

	# Filter out close atoms in the gb again; possible to have anamolies due to multiple grain overlap
	cutoff = kwargs.get('cutoff',0.2)
	if len(gbPositions) > 0:
		closeAtoms = True
		while closeAtoms:
			# Find minDist with c routine
			minInd = numpy.zeros(len(gbPositions)-1)
			minDist = numpy.zeros(len(gbPositions)-1)
			_cPolyUtils.selfClosest(L[0],L[1],gbPositions[:,0],gbPositions[:,1],minInd,minDist)

			gbPositions = numpy.delete(gbPositions,numpy.where(minDist < cutoff*a),axis=0)

			minInd = numpy.zeros(len(gbPositions)-1)
			minDist = numpy.zeros(len(gbPositions)-1)
			_cPolyUtils.selfClosest(L[0],L[1],gbPositions[:,0],gbPositions[:,1],minInd,minDist)
			if minDist.min() >= cutoff*a:
				closeAtoms = False

	# collect all positions
	positions = numpy.concatenate((bulkPositions,unRatPositions))
	positions = numpy.concatenate((positions,gbPositions))
	positions = numpy.concatenate((positions,padPositions))
	tags = numpy.zeros(len(positions))
	tags[-len(unRatPositions)-len(gbPositions)-len(padPositions):] = 3
	tags[-len(gbPositions)-len(padPositions):] = 1
	tags[-len(padPositions):] = 2

	# Set the gbPositions as rattlable, and then merge the unRat and gb
	rattlable = numpy.zeros(len(tags))
	rattlable[tags == 1] = 1
	tags[tags==3] = 1


	cell = numpy.diag([L[0],L[1],10])
	#cr = Atoms(symbols=symbols, positions=positions, cell = cell, pbc=[True,True,True])
	numbers = numpy.ones(len(positions))*6
	symbols = ['C']*len(positions)
	cr = Atoms(numbers = numbers, positions=positions, cell = cell, tags=tags,pbc=[True,True,True])
	cr.ax = ax
	cr.centers = centers
	cr.orgs = orgs
	cr.rattlable = rattlable
	if sum(tags == 0) == 0 or sum(tags ==1 ) ==0 or sum(tags == 2) == 0:
		print "It seems that either the cell is too small or there are too many grains. You might want to fix this"
	return cr


def cvtRelax(gr,**kwargs):
	"""
	Relax a triangular lattice by the cvt algorithm
	"""
	positions = gr.positions.copy()

	if kwargs.get('rattle',False):	# Rattle 
		ratMax = kwargs.get('ratMax',1.0)
		positions[gr.rattlable==1,0] += ((numpy.random.uniform(size=sum(gr.rattlable)) -0.5))*ratMax
		positions[gr.rattlable==1,1] += ((numpy.random.uniform(size=sum(gr.rattlable)) -0.5))*ratMax

	repeat = True
	bulk = positions[gr.get_tags()==0,0:2]		# Bulk points
	pad = positions[gr.get_tags()==2,0:2]		# Pad points
	bdry = positions[gr.get_tags()==1,0:2]		# boundary points

	maxIter = kwargs.get('maxIter',10000)
	tol = kwargs.get('tol',1E-5)
	verbose = kwargs.get('verbose',False)
	itNum = 0

	# periodic pad points 
	# we only take periodic image of a strip of width sw (strip width)
	sw = kwargs.get('sw',10.0)
	ix = numpy.array([[-1,-1,-1,0,0,1,1,1],[-1,0,1,-1,1,-1,0,1]])
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
		_cPolyUtils.vorCentroid(len(Z) - len(bdry), len(Z),vor.regions,vor.point_region,vor.points[:,0],vor.points[:,1],vor.vertices[:,0],vor.vertices[:,1],ox,oy)

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
	ix = numpy.indices((3,3), dtype=int).reshape(2,-1) -1
	tvecs = numpy.einsum('ki,kj',ix,numpy.diag(L))
	cents = numpy.tile(tvecs.T,len(Z)).T + Z.repeat(9,axis=0)	# centers
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
		_cPolyUtils.selfClosest(tr.cell[0,0],tr.cell[1,1],bPos[:,0],bPos[:,1],minInd,minDist)

		dist = numpy.ones(len(pos))*2*cutoff
		dist[boundary[:-1]==1] = minDist
		closeFlag[dist<cutoff] = 1

		tags[boundary==1] = 2

	pos = numpy.delete(pos,numpy.where(closeFlag == 1),axis=0)
	tags = numpy.delete(tags,numpy.where(closeFlag == 1),axis=0)

	# Check
	"""
	# Find minDist with c routine
	minInd = numpy.zeros(len(pos)-1)
	minDist = numpy.zeros(len(pos)-1)
	_cPolyUtils.selfClosest(tr.cell[0,0],tr.cell[1,1],pos[:,0],pos[:,1],minInd,minDist)
	if (minDist < cutoff).any():
		print "Error"
	"""

	numbers = [num[x] for x in tags]
	pos[:,2] = tr.cell[2,2]*0.5
	cr = ase.Atoms(numbers=numbers,positions=pos,cell=tr.cell,tags=tags,pbc=[True,True,True])

	return cr

		
