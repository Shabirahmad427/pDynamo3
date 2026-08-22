"""The PySCF QC model."""

import glob, os

try:
    import pyscf
    _PYSCFFound = True
except:
    _PYSCFFound = False

# . Availability is checked WITHOUT importing gpu4pyscf here (find_spec does
#   not execute the module). This matters because importing gpu4pyscf touches
#   cupy/CUDA, and CUDA contexts are not fork-safe: EasyHybrid3 runs each
#   simulation in a forked worker process (multiprocessing default on Linux),
#   and a worker that inherits an already-initialized CUDA context from the
#   parent fails with "cudaErrorInitializationError: initialization error"
#   the moment it tries to use the GPU. The real "import gpu4pyscf" happens
#   lazily in CreateMole() below, every time an energy is actually computed
#   -- which always runs inside the (freshly forked) worker process, never
#   in the parent GUI process, so the parent's CUDA context stays untouched.
import importlib.util
_GPU4PySCFFound = importlib.util.find_spec ( 'gpu4pyscf' ) is not None

from pCore             import NotInstalledError
from pMolecule.QCModel import QCModel           , \
                              QCModelError
from pScientific       import Units

#===================================================================================================================================
# . Parameters.
#===================================================================================================================================
_DefaultFunctional = "blyp"

#===================================================================================================================================
# . Class.
#===================================================================================================================================
class QCModelPySCF ( QCModel ):
    """The PySCF QC model class."""

    _attributable = dict ( QCModel._attributable )
    _classLabel   = "PySCF QC Model"
    _summarizable = dict ( QCModel._summarizable )
    _unpicklable  = set ( QCModel._unpicklable )
    # . "pyscf" holds a live reference to the imported pyscf module, and
    #   "mf"/"mole" hold live PySCF calculation objects (built in
    #   CreateMole) -- none of these can be pickled (e.g. a system with this
    #   QCModel attached fails "TypeError: cannot pickle 'module' object" on
    #   Pickle()/multiprocessing.Process(), both of which pickle the target
    #   system, including its qcModel). _CheckOptions() and CreateMole()
    #   rebuild them from scratch every time regardless, so excluding them
    #   from the pickled state is safe.
    # . "gpu4pyscf" is a live module reference too (same reason as "pyscf").
    _unpicklable.update ( { "gpu4pyscf", "mf", "mole", "pyscf" } )
    _attributable.update ( { "deleteJobFiles" : True    ,
                             "functional"     : "blyp"  ,
                             "gpu4pyscf"      : None    ,
                             "method"         : "RHF"   ,
                             "mf"             : None    ,
                             "mf_kwargs"      : dict    ,
                             "mole"           : None    ,
                             "mole_kwargs"    : dict    ,
                             "orbitalBasis"   : "3-21G" ,
                             "pyscf"          : None    ,
                             "pySCFscratch"   : None    ,
                             "useGPU"         : False   } )
    _summarizable.update ( { "functional"     : "Functional"    ,
                             "method"         : "Method"        ,
                             "orbitalBasis"   : "Orbital Basis" ,
                             "useGPU"         : "Run On GPU"    } )

    def __del__ ( self ):
        """Deallocation."""
        self.DeleteJobFiles ( )

    def _CheckOptions ( self ):
        """Check options."""
        super ( QCModelPySCF, self )._CheckOptions ( )
        # . pySCF.
        if _PYSCFFound: self.pyscf = pyscf
        else: raise NotInstalledError ( "pySCF not installed." )
        # . gpu4pyscf (only required if useGPU is set). Availability only --
        #   the real import happens lazily in CreateMole() (see the comment
        #   by _GPU4PySCFFound above for why).
        if self.useGPU and not _GPU4PySCFFound:
            raise NotInstalledError ( "gpu4pyscf not installed." )
        # . pySCFscratch.
        scratch = os.getenv ( 'PYSCF_TMPDIR' )
        if scratch is None:
            scratch                    = os.getenv ( "PDYNAMO3_SCRATCH" )
            os.environ['PYSCF_TMPDIR'] = str ( scratch )
        self.pySCFscratch = scratch
        # . Functional.
        if ( 'KS' in self.method.upper ( ) ) and ( self.functional is None ):
            self.functional = _DefaultFunctional

    def CreateMole ( self, target, doQCMM ):
        """Create PySCF mole and mean-field objects"""
        state        = getattr ( target, self.__class__._stateName )
        n            = len ( state.atomicNumbers )
        coordinates3 = target.scratch.qcCoordinates3AU
        mole         = self.pyscf.gto.Mole()
        mole.atom    = [[state.atomicNumbers[i], (coordinates3[i][0], coordinates3[i][1], coordinates3[i][2])] for i in range(n)]
        mole.basis   = self.orbitalBasis
        mole.charge  = target.electronicState.charge
        mole.spin    = target.electronicState.multiplicity - 1 # 2S
        mole.unit    = 'Bohr'
        mole.verbose = 0
        mole.__dict__.update(self.mole_kwargs)
        state.mole = mole.build()
        if self.useGPU:
            # . Import here, not at module level -- see the comment by
            #   _GPU4PySCFFound near the top of this file.
            import gpu4pyscf
            import gpu4pyscf.qmmm
            # Force cuTENSOR engine (CuPy detects it, but gpu4pyscf's
            # detection sometimes fails with newer cuTENSOR versions)
            from gpu4pyscf.lib import cutensor
            cutensor.contract_engine = 'cutensor'
            self.gpu4pyscf = gpu4pyscf
            # . gpu4pyscf has no Mole.apply()-style dispatcher (that's a
            #   pyscf.gto.Mole convenience method); build the mean-field
            #   object directly from its GPU-accelerated RHF/UHF/RKS/UKS
            #   classes instead, applying mf_kwargs as attributes the same
            #   way Mole.apply() does internally.
            gpuMethods = { "RHF" : self.gpu4pyscf.scf.RHF ,
                          "UHF" : self.gpu4pyscf.scf.UHF ,
                          "RKS" : self.gpu4pyscf.dft.RKS ,
                          "UKS" : self.gpu4pyscf.dft.UKS }
            methodClass = gpuMethods.get ( self.method.upper ( ), None )
            if methodClass is None:
                raise QCModelError ( "gpu4pyscf only supports the RHF, UHF, RKS and UKS methods." )
            state.mf = methodClass ( state.mole )
            for ( key, value ) in self.mf_kwargs.items ( ): setattr ( state.mf, key, value )
        else:
            state.mf = state.mole.apply ( self.method, **self.mf_kwargs )
        if doQCMM:
            charges       = []
            chargesB      = getattr ( target.qcmmState, "bpCharges", None )
            chargesM      = target.mmState.charges
            coords        = []
            coordinates3B = target.scratch.Get ( "bpCoordinates3", None                ) 
            coordinates3M = target.scratch.Get ( "coordinates3NB", target.coordinates3 )
            mmAtoms       = target.mmState.pureMMAtoms
            qScale        = 1.0 / target.qcmmElectrostatic.dielectric
            nM            = len ( mmAtoms )
            if chargesB is None: nB = 0
            else:                nB = len ( chargesB )
            for i in mmAtoms:
               charges.append(qScale * chargesM[i])
               coords.append ([ coordinates3M[i,0], coordinates3M[i,1], coordinates3M[i,2] ])
            for i in range ( nB ):
               charges.append(qScale * chargesB[i])
               coords.append ([ coordinates3B[i,0], coordinates3B[i,1], coordinates3B[i,2] ])
            # . Coordinates for MM centers stored in Angstroms.
            qmmmModule = self.gpu4pyscf.qmmm if self.useGPU else self.pyscf.qmmm
            state.mf = qmmmModule.add_mm_charges(state.mf, coords, charges, unit='Angstrom')

    def DeleteJobFiles ( self ):
        """Delete job files."""
        if self.deleteJobFiles:
            try:
                jobFiles = glob.glob ( os.path.join ( self.pySCFscratch, "tmp????????" ) )
                for jobFile in jobFiles: os.remove ( jobFile )
            except:
                pass

    def Energy ( self, target ):
        """Calculate the quantum chemical energy and gradient."""
        # . "pyscf" (and "mf"/"mole") are excluded from pickling (see
        #   _unpicklable above), so after Unpickle() or a multiprocessing
        #   handoff self.pyscf is None here -- re-derive it (re-"import
        #   pyscf") before use.
        if self.pyscf is None: self._CheckOptions ( )
        state           = getattr ( target, self.__class__._stateName )
        xc              = None
        if 'KS' in self.method.upper(): xc = self.functional
        doGradients     = target.scratch.doGradients
        doQCMM          = ( len ( target.atoms ) > len ( state.qcAtoms ) )
        try: 
            self.CreateMole ( target, doQCMM )
        except:
            raise QCModelError ( "Error creating PySCF mole and mean-field objects." )

        # . Calculate energy.
        try:     
            state.mf.run(xc=xc)
        except:
           raise QCModelError ( "Error calculating PySCF energy." )

        if not state.mf.converged: raise QCModelError ( "SCF energy calculation did not converge." )
        target.scratch.energyTerms["PySCF QC"] = ( state.mf.energy_tot() * Units.Energy_Hartrees_To_Kilojoules_Per_Mole )
        if doGradients:
            try: 
                g  = state.mf.nuc_grad_method()
                ga = g.kernel()
            except:
                raise QCModelError ( "Error calculating PySCF gradient." )  
            # . QC gradients
            for i in range ( len ( state.atomicNumbers ) ):
                for j in range ( 3 ):
                   target.scratch.qcGradients3AU[i,j] = ga[i,j]

#===================================================================================================================================
# . Testing.
#===================================================================================================================================
if __name__ == "__main__" :
    pass

