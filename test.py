import grainBdr as gb
import polyCrystal as pc
import ase.io
import numpy

cr = gb.onePeriodicGB(N1=[1,2],N2=[2,1],cell_width=100,verbose=False)
ase.io.write('testGB_1Periodic.pdb',cr)
ase.io.write('testGB_1Periodic.cfg',cr)
gb.writeLammpsData(cr,'testGB_1Periodic.lammps')


cr = gb.twoPeriodicGB(N1=[1,2],N2=[2,1],cell_width=200,verbose=False)
ase.io.write('testGB_2Periodic.pdb',cr)
ase.io.write('testGB_2Periodic.cfg',cr)
gb.writeLammpsData(cr,'testGB_2Periodic.lammps')

cr = pc.periodicCrystal(L=numpy.array([100,100]),N=4)
ase.io.write('testPoly_2Periodic.pdb',cr)
ase.io.write('testPoly_2Periodic.cfg',cr)
gb.writeLammpsData(cr,'testPoly_2Periodic.lammps')

