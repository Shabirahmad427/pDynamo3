"""The PySCF QC model."""

import glob, os, tempfile

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
    _unpicklable.update({"_scratchDirectory", "_scratchPID"})
    # . Speed options for repeated calculations (MD, scans), following
    #   QM_eMMa's production GPU4PySCF settings. All are off by default so
    #   existing results do not change:
    #   densityFitting - RI/density fitting of the Coulomb (and exchange) terms.
    #   auxiliaryBasis - auxiliary basis for densityFitting (None = PySCF default).
    #   gridLevel      - DFT integration grid level (None = PySCF default, 3).
    #   reuseDensity   - start each SCF from the previous converged density
    #                    (same result, fewer cycles); falls back to a cold
    #                    start if the warm SCF does not converge.
    _attributable.update ( { "auxiliaryBasis" : None    ,
                             "densityFitting" : False   ,
                             "gridLevel"      : None    ,
                             "reuseDensity"   : False   } )
    _summarizable.update ( { "densityFitting" : "Density Fitting" ,
                             "gridLevel"      : "Grid Level"      ,
                             "reuseDensity"   : "Reuse Density"   } )
    _attributable.update ( { "deleteJobFiles" : True    ,
                             "functional"     : "blyp"  ,
                             "gpu4pyscf"      : None    ,
                             "method"         : "RHF"   ,
                             "mf"             : None    ,
                             "mf_kwargs"      : dict    ,
                             "mole"           : None    ,
                             "mole_kwargs"    : dict    ,
                             "orbitalBasis"   : "3-21G" ,
                             "pm6AtomicNumbers"  : None ,
                             "pm6EnergyBaseLine" : None ,
                             "pyscf"          : None    ,
                             "pySCFscratch"   : None    ,
                             "_scratchDirectory": None,
                             "_scratchPID": None,
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
        method = self.method.upper ( )
        if ( method == "PM6" ) and ( not self.useGPU ):
            raise QCModelError ( "GPU4PySCF PM6 requires the Run On GPU option." )
        # . pySCF.
        if _PYSCFFound: self.pyscf = pyscf
        else: raise NotInstalledError ( "pySCF not installed." )
        # . gpu4pyscf (only required if useGPU is set). Availability only --
        #   the real import happens lazily in CreateMole() (see the comment
        #   by _GPU4PySCFFound above for why).
        if self.useGPU and not _GPU4PySCFFound:
            raise NotInstalledError ( "gpu4pyscf not installed." )
        # . pySCFscratch.
        scratch = self.pySCFscratch or os.getenv ( 'PYSCF_TMPDIR' )
        if scratch is None:
            scratch                    = os.getenv ( "PDYNAMO3_SCRATCH" )
            os.environ['PYSCF_TMPDIR'] = str ( scratch )
        if not scratch:
            raise QCModelError("No PySCF scratch directory is configured.")
        self.pySCFscratch = os.path.abspath(os.path.expanduser(scratch))
        # . Functional.
        if ( 'KS' in self.method.upper ( ) ) and ( self.functional is None ):
            self.functional = _DefaultFunctional

    def CreateMole ( self, target, doQCMM ):
        """Create PySCF mole and mean-field objects"""
        # PySCF reads this runtime setting when creating checkpoint files.
        # Set it for the active model, including after switching systems.
        self.pyscf.lib.param.TMPDIR = self._GetScratchDirectory()
        state        = getattr ( target, self.__class__._stateName )
        n            = len ( state.atomicNumbers )
        coordinates3 = target.scratch.qcCoordinates3AU
        method       = self.method.upper ( )

        # GPU4PySCF's semiempirical Mole/RHF classes are independent of the
        # Gaussian-basis PySCF classes used below.  In particular, the stock
        # gpu4pyscf.qmmm wrapper evaluates Gaussian int1e_grids integrals and
        # is not compatible with sem.Mole.  Reject embedded calculations
        # explicitly: running the isolated QM region would give a plausible
        # looking but physically incorrect QM/MM energy and gradient.
        if method == "PM6":
            if doQCMM:
                raise QCModelError ( "GPU4PySCF PM6 does not currently support pDynamo QM/MM embedding." )
            if not self.useGPU:
                raise QCModelError ( "GPU4PySCF PM6 requires the Run On GPU option." )
            if target.electronicState.multiplicity != 1:
                raise QCModelError ( "GPU4PySCF PM6 currently supports restricted closed-shell singlets only." )
            import gpu4pyscf
            from gpu4pyscf.lib          import cutensor
            from gpu4pyscf.sem.gto.mole import Mole as SemiempiricalMole
            from gpu4pyscf.sem.scf.hf   import RHF  as SemiempiricalRHF
            from pMolecule.QCModel.MNDOParametersContainer import MNDOParametersContainer
            cutensor.contract_engine = 'cutensor'
            self.gpu4pyscf = gpu4pyscf
            atoms = [ [ state.atomicNumbers[i],
                        ( coordinates3[i][0], coordinates3[i][1], coordinates3[i][2] ) ]
                      for i in range ( n ) ]
            options = dict ( self.mole_kwargs )
            options.update ( { "atom"   : atoms,
                               "charge" : target.electronicState.charge,
                               "method" : "PM6",
                               "spin"   : target.electronicState.multiplicity - 1,
                               "unit"   : "Bohr" } )
            state.mole = SemiempiricalMole ( **options ).build ( )
            state.mf   = SemiempiricalRHF  ( state.mole )
            # GPU4PySCF reports the raw PM6 electronic+nuclear energy.  Native
            # pDynamo PM6 adds this atom-dependent constant so its result uses
            # the heat-of-formation convention.  Preserve that convention so
            # GPU and CPU energies can be compared in EasyHybrid projects.
            atomicNumbers = tuple ( int ( z ) for z in state.atomicNumbers )
            if self.pm6AtomicNumbers != atomicNumbers:
                parameters = MNDOParametersContainer.FromParameterDirectory ( "pm6", state.atomicNumbers )
                self.pm6AtomicNumbers  = atomicNumbers
                self.pm6EnergyBaseLine = parameters.energyBaseLine
            for ( key, value ) in self.mf_kwargs.items ( ): setattr ( state.mf, key, value )
            return

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
            methodClass = gpuMethods.get ( method, None )
            if methodClass is None:
                raise QCModelError ( "gpu4pyscf only supports the RHF, UHF, RKS and UKS methods." )
            state.mf = methodClass ( state.mole )
            for ( key, value ) in self.mf_kwargs.items ( ): setattr ( state.mf, key, value )
        else:
            state.mf = state.mole.apply ( self.method, **self.mf_kwargs )
        # . Density fitting and grid level are applied before the QM/MM
        #   wrapper, as in QM_eMMa (RKS -> density_fit -> mm charges).
        if self.densityFitting:
            state.mf = state.mf.density_fit ( auxbasis = self.auxiliaryBasis )
        if ( self.gridLevel is not None ) and hasattr ( state.mf, "grids" ):
            state.mf.grids.level = int ( self.gridLevel )
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

    def _GetScratchDirectory(self):
        """Give each model and worker process its own temporary directory."""
        if self._scratchDirectory is None or self._scratchPID != os.getpid():
            os.makedirs(self.pySCFscratch, exist_ok=True)
            self._scratchDirectory = tempfile.mkdtemp(prefix='pyscf-job-', dir=self.pySCFscratch)
            self._scratchPID = os.getpid()
        return self._scratchDirectory

    def DeleteJobFiles ( self ):
        """Delete job files."""
        if (self.deleteJobFiles and getattr(self, '_scratchDirectory', None)
                and self._scratchPID == os.getpid()):
            try:
                jobFiles = glob.glob ( os.path.join ( self._scratchDirectory, "tmp????????" ) )
                for jobFile in jobFiles: os.remove ( jobFile )
                if not os.listdir(self._scratchDirectory):
                    os.rmdir(self._scratchDirectory)
                    self._scratchDirectory = None
                    self._scratchPID = None
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
        method          = self.method.upper ( )
        if self.useGPU and ( getattr ( target, "symmetry", None ) is not None or
                             getattr ( target, "symmetryParameters", None ) is not None ):
            raise QCModelError ( "GPU4PySCF supports molecular calculations only; periodic boundary conditions are not implemented." )
        if self.useGPU and target.electronicState.isSpinRestricted and target.electronicState.multiplicity != 1:
            raise QCModelError ( "GPU4PySCF restricted open-shell calculations are not supported. Select Unrestricted spin mode." )
        xc              = None
        if 'KS' in method: xc = self.functional
        doGradients     = target.scratch.doGradients
        doQCMM          = ( len ( target.atoms ) > len ( state.qcAtoms ) )
        try:
            self.CreateMole ( target, doQCMM )
        except Exception as error:
            raise QCModelError ( "Error creating PySCF molecule and mean-field objects: {:s}".format ( str ( error ) ) ) from error

        # . Calculate energy.
        #   With reuseDensity, the previous converged density is the starting
        #   guess when it belongs to the same electronic problem (atoms, charge,
        #   multiplicity, method, basis, functional, density fitting).
        #   mf.kernel ( dm0 = None ) is what mf.run ( ) does, so with no stored
        #   density this is the original calculation.
        densityKey = ( method, self.orbitalBasis, xc, tuple ( int ( z ) for z in state.atomicNumbers ),
                       target.electronicState.charge, target.electronicState.multiplicity,
                       bool ( self.densityFitting ), self.auxiliaryBasis )
        dm0 = None
        if self.reuseDensity and ( method != "PM6" ) and ( getattr ( self, "_previousDensityKey", None ) == densityKey ):
            dm0 = getattr ( self, "_previousDensity", None )
        self.lastDensityGuess = "cold" if dm0 is None else "previous"
        try:
            if method == "PM6": state.mf.run ( )
            else:
                if xc is not None: state.mf.xc = xc
                state.mf.kernel ( dm0 = dm0 )
                if ( dm0 is not None ) and ( not state.mf.converged ):
                    # . Warm start failed: rebuild and start from the default guess.
                    self.CreateMole ( target, doQCMM )
                    if xc is not None: state.mf.xc = xc
                    state.mf.kernel ( )
                    self.lastDensityGuess = "previous_then_cold"
        except Exception as error:
           self._previousDensity = self._previousDensityKey = None
           raise QCModelError ( "Error calculating PySCF energy: {:s}".format ( str ( error ) ) ) from error
        self.lastSCFCycles = getattr ( state.mf, "cycles", None )

        if not state.mf.converged:
            self._previousDensity = self._previousDensityKey = None
            raise QCModelError ( "SCF energy calculation did not converge." )
        if self.reuseDensity and ( method != "PM6" ):
            self._previousDensity    = state.mf.make_rdm1 ( )
            self._previousDensityKey = densityKey
        energy = state.mf.energy_tot ( )
        if method == "PM6": energy += self.pm6EnergyBaseLine
        label = "GPU4PySCF PM6" if method == "PM6" else "PySCF QC"
        target.scratch.energyTerms[label] = ( energy * Units.Energy_Hartrees_To_Kilojoules_Per_Mole )
        if doGradients:
            try:
                if method == "PM6": g = state.mf.Gradients ( )
                else:                g = state.mf.nuc_grad_method ( )
                ga = g.kernel()
            except Exception as error:
                raise QCModelError ( "Error calculating PySCF gradient: {:s}".format ( str ( error ) ) ) from error
            # . QC gradients
            for i in range ( len ( state.atomicNumbers ) ):
                for j in range ( 3 ):
                   target.scratch.qcGradients3AU[i,j] = ga[i,j]

#===================================================================================================================================
# . Testing.
#===================================================================================================================================
if __name__ == "__main__" :
    pass
