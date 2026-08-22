"""Defines a QC/MM electrostatic model compatible with the PySCF program."""

from pCore             import NotInstalledError
from pMolecule.NBModel import QCMMElectrostaticModel
from pScientific       import Units

try:
    import numpy
except:
    raise NotInstalledError ( "numpy not installed." )

try:
    import cupy
    def _AsNumpy ( array ):
        """gpu4pyscf's mean-field objects (target.qcState.mf when useGPU is
        set on QCModelPySCF) return cupy (GPU-resident) arrays from methods
        like mm_mol.atom_coords()/make_rdm1(); the rest of this module does
        plain numpy math, so bring anything cupy back to host memory first.
        A no-op for regular pyscf's numpy arrays."""
        return cupy.asnumpy ( array ) if isinstance ( array, cupy.ndarray ) else array
except ImportError:
    def _AsNumpy ( array ):
        return array

#===================================================================================================================================
# . Class.
#===================================================================================================================================
class QCMMElectrostaticModelPySCF ( QCMMElectrostaticModel ):

    _classLabel  = "PySCF QC/MM Electrostatic Model"

    def QCMMGradients ( self, target ):
        # . This works for mean-field (R/U)(HF/KS) methods. For post-HF methods, the orbital response needs to be implemented.
        # . Gradients for ROHF with qmmm charges do not work in PySCF (1.7.0).
        # . gradients3B/M are handled by NBModel.PairwiseInteraction* in pyrex/csource.
        # . The gradients are in atomic units!
        if target.scratch.doGradients:
               dmi        = target.qcState.mf.make_rdm1 ( )
               if len(dmi.shape) > 2: 
                   dm = dmi[0] + dmi[1] # . Unrestricted/RestrictedOpen-shell DM, so we sum alpha and beta contribs.
               else: 
                   dm = dmi
               # . Use PySCF's blocked MM-center gradient implementation.
               #   The former scalar loop evaluated one integral per MM center
               #   and dominated relaxed PES calculations for large systems.
               #   Both PySCF and GPU4PySCF expose these two methods; the GPU
               #   implementation returns the contracted result on the host.
               gradientModel = target.qcState.mf.nuc_grad_method ( )
               g = _AsNumpy ( gradientModel.grad_hcore_mm ( dm ) )
               g += _AsNumpy ( gradientModel.grad_nuc_mm ( ) )

               n = len (g)
               if n > 0:
                  factor      = Units.Length_Angstroms_To_Bohrs * Units.Energy_Hartrees_To_Kilojoules_Per_Mole
                  gradients3B = target.scratch.Get ( "bpGradients3", None )
                  gradients3M = target.scratch.gradients3
                  mmAtoms     = target.mmState.pureMMAtoms
                  nM          = len ( mmAtoms )
                  for i in range ( n ):
                      if i < nM:
                          s = mmAtoms[i]
                          for j in range(3): gradients3M[s,j] = g[i][j] * factor
                      else:
                          s = i-nM
                          for j in range(3): gradients3B[s,j] = g[i][j] * factor

#===================================================================================================================================
# . Testing.
#===================================================================================================================================
if __name__ == "__main__" :
    pass
