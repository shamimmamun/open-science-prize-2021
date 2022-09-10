from qiskit.providers.aer import QasmSimulator

# load IBMQ Account data
from qiskit import IBMQ
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit.ignis.verification.tomography import state_tomography_circuits
from qiskit.result import Result

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.result.counts import Counts


from collections import Counter
from functools import reduce
from qiskit import assemble, execute, transpile
import numpy as np

# Noiseless simulated backend
sim = QasmSimulator()


IBMQ.save_account("TOKEN", overwrite=True) 
account = IBMQ.load_account()

provider = IBMQ.get_provider(hub='ibm-q-community', group='ibmquantumawards', project='open-science-22')
jakarta = provider.get_backend('ibmq_jakarta')

# Simulated backend based on ibmq_jakarta's device noise profile
sim_noisy_jakarta = QasmSimulator.from_backend(provider.get_backend('ibmq_jakarta'))



# Trotterized circuit from IBM material


def get_circuit(measure, trotter_steps, X=True, Y=True, Z=True):

    # YOUR TROTTERIZATION GOES HERE -- START (beginning of example)

    # Parameterize variable t to be evaluated at t=pi later
    t = Parameter('t')

    # Build a subcircuit for XX(t) two-qubit gate
    XX_qr = QuantumRegister(2)
    XX_qc = QuantumCircuit(XX_qr, name='XX')

    XX_qc.ry(np.pi/2,[0,1])
    XX_qc.cnot(0,1)
    XX_qc.rz(2 * t, 1)
    XX_qc.cnot(0,1)
    XX_qc.ry(-np.pi/2,[0,1])

    # Convert custom quantum circuit into a gate
    XX = XX_qc.to_instruction()

    # Build a subcircuit for YY(t) two-qubit gate
    YY_qr = QuantumRegister(2)
    YY_qc = QuantumCircuit(YY_qr, name='YY')

    YY_qc.rx(np.pi/2,[0,1])
    YY_qc.cnot(0,1)
    YY_qc.rz(2 * t, 1)
    YY_qc.cnot(0,1)
    YY_qc.rx(-np.pi/2,[0,1])

    # Convert custom quantum circuit into a gate
    YY = YY_qc.to_instruction()

    # Build a subcircuit for ZZ(t) two-qubit gate
    ZZ_qr = QuantumRegister(2)
    ZZ_qc = QuantumCircuit(ZZ_qr, name='ZZ')

    ZZ_qc.cnot(0,1)
    ZZ_qc.rz(2 * t, 1)
    ZZ_qc.cnot(0,1)

    # Convert custom quantum circuit into a gate
    ZZ = ZZ_qc.to_instruction()

    # Combine subcircuits into a single multiqubit gate representing a single trotter step
    num_qubits = 3

    Trot_qr = QuantumRegister(num_qubits)
    Trot_qc = QuantumCircuit(Trot_qr, name='Trot')

    for i in range(0, num_qubits - 1):
        if Z:
            Trot_qc.append(ZZ, [Trot_qr[i], Trot_qr[i+1]])
        
        if Y:
            Trot_qc.append(YY, [Trot_qr[i], Trot_qr[i+1]])
        
        if X:
            Trot_qc.append(XX, [Trot_qr[i], Trot_qr[i+1]])

    # Convert custom quantum circuit into a gate
    Trot_gate = Trot_qc.to_instruction()
    
    # YOUR TROTTERIZATION GOES HERE -- FINISH (end of example)


    # The final time of the state evolution
    target_time = np.pi

    # Number of trotter steps
    #trotter_steps = 8  ### CAN BE >= 4

    # Initialize quantum circuit for 3 qubits
    qr = QuantumRegister(7)
    cr = ClassicalRegister(3)
    qc = QuantumCircuit(qr, cr) if measure is True else QuantumCircuit(qr)

    # Prepare initial state (remember we are only evolving 3 of the 7 qubits on jakarta qubits (q_5, q_3, q_1) corresponding to the state |110>)
    qc.x([3,5])  # DO NOT MODIFY (|q_5,q_3,q_1> = |110>)

    # Simulate time evolution under H_heis3 Hamiltonian
    for _ in range(trotter_steps):
        qc.append(Trot_gate, [qr[1], qr[3], qr[5]])
        if not X or not Y or not Z:
            break
    
        

    # Evaluate simulation at target_time (t=pi) meaning each trotter step evolves pi/trotter_steps in time
    qc = qc.bind_parameters({t: target_time/trotter_steps})

    if measure:
        qc.measure([1,3,5], cr)

    
    return qc




trotters = 12

train_st_qcs_xy = state_tomography_circuits(get_circuit(False, trotters, True, True, False), [1,3,5])
train_st_qcs_yz = state_tomography_circuits(get_circuit(False, trotters, False, True, True), [1,3,5])
train_st_qcs_zx = state_tomography_circuits(get_circuit(False, trotters, True, False, True), [1,3,5])





def sorted_counts(counts):
    complete = dict(reduce(lambda a, b: a.update(b) or a, [{'000': 0, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 0}, counts], Counter()))
    return {k: v for k, v in sorted(complete.items(), key=lambda item: item[0])}


def get_modifiers(qc, shots = 4096, display=True):
    
    t_qc_sim = transpile(qc, sim)
    noiseless_result = sim.run( assemble(t_qc_sim), shots=shots).result()
    noiseless_counts = sorted_counts(noiseless_result.get_counts())
    
    t_qc = transpile(qc, sim_noisy_jakarta)
    qobj = assemble(t_qc)
    counts = sorted_counts(sim_noisy_jakarta.run(qobj, shots=shots).result().get_counts())
    
    zipped = list(zip(noiseless_counts.values(), counts.values()))
    modifier = list(map(lambda pair: pair[0]/pair[1] if pair[1] > 0 else 1, zipped))

    if display is True:
        print("noisy:     ", counts)
        print("nose-free: ", noiseless_counts)

        print("modifier: ", modifier)
        print("\n")
    
    return modifier




modifiers_xy = list(map(lambda qc: get_modifiers(qc, display=False), train_st_qcs_xy))
modifiers_yz = list(map(lambda qc: get_modifiers(qc, display=False), train_st_qcs_yz))
modifiers_zx = list(map(lambda qc: get_modifiers(qc, display=False), train_st_qcs_zx))
	
modifiers_zipped = list(zip(modifiers_xy, modifiers_yz, modifiers_zx))
	
def mult(tup):
	zipped = zip(*tup)
	return list(map(lambda x: (x[0]*x[1]*x[2])**(trotters/2), zipped))
	
mods = list(map(mult, modifiers_zipped))



class OwnResult(Result):
    
    def __init__(self, result):
        self._result = result
        self._counts = {}
            
        
    def get_counts(self, experiment=None):

        if experiment is None:
            exp_keys = range(len(self._result.results))
        else:
            exp_keys = [experiment]
        

        dict_list = []
        for key in exp_keys:
            exp = self._result._get_experiment(key)
            try:
                header = exp.header.to_dict()
            except (AttributeError, QiskitError):  # header is not available
                header = None

            if "counts" in self._result.data(key).keys():
                if header:
                    counts_header = {
                        k: v
                        for k, v in header.items()
                        if k in {"time_taken", "creg_sizes", "memory_slots"}
                    }
                else:
                    counts_header = {}
                    
                    
                # CUSTOM CODE STARTS HERE #######################
                dict_list.append(Counts(
                    self._counts[str(key)] if str(key) in map(lambda k: str(k), self._counts.keys()) else self._result.data(key)["counts"]
                    , **counts_header))
                # CUSTOM CODE ENDS HERE #######################
                
            elif "statevector" in self._result.data(key).keys():
                vec = postprocess.format_statevector(self._result.data(key)["statevector"])
                dict_list.append(statevector.Statevector(vec).probabilities_dict(decimals=15))
            else:
                raise QiskitError(f'No counts for experiment "{repr(key)}"')

        # Return first item of dict_list if size is 1
        if len(dict_list) == 1:
            return dict_list[0]
        else:
            return dict_list
        
        
    def set_counts(self, counts, experiment=None):
        self._counts[str(experiment) if experiment is not None else "0"] = counts



   
# Compute the state tomography based on the st_qcs quantum circuits and the results from those ciricuits
def state_tomo(result, st_qcs, modifiers, mitigate=False):
    # The expected final state; necessary to determine state tomography fidelity
    target_state = (One^One^Zero).to_matrix()  # DO NOT MODIFY (|q_5,q_3,q_1> = |110>)
    
    own_res = OwnResult(result)
    
    idx = 0
    
    if mitigate:
        for experiment in st_qcs:
            exp_keys = [experiment]
            for key in exp_keys:

                exp = result._get_experiment(key)                
                counts = sorted_counts(result.get_counts(key))
                mitigated = {item[0]: item[1]*modifiers[idx][i] for i, item in enumerate(counts.items())}
                
                #print("original: ", sorted_counts(result.get_counts(key)))
                #print("mitigated: ", sorted_counts(mitigated))
                #print("\n")

                own_res.set_counts(mitigated, key)
    
            idx = idx + 1 
        
    
    # Fit state tomography results
    tomo_fitter = StateTomographyFitter(own_res if mitigate else result, st_qcs)
    rho_fit = tomo_fitter.fit(method='lstsq')
    # Compute fidelity
    fid = state_fidelity(rho_fit, target_state)
    return fid



shots=4096

st_qcs = state_tomography_circuits(get_circuit(False, trotters, True, True, True), [1,3,5])

noisy_job = execute(st_qcs, sim_noisy_jakarta, shots=shots)
noisefree_job = execute(st_qcs, sim, shots=shots)
    
noisy_fid = state_tomo(noisy_job.result(), st_qcs, mods, mitigate=False)
noisefree_fid = state_tomo(noisefree_job.result(), st_qcs, mods, mitigate=False)
mitigated_fid = state_tomo(noisy_job.result(), st_qcs, mods, mitigate=True)

print('noisy state tomography fidelity = {:.4f} \u00B1 {:.4f}'.format(np.mean([noisy_fid]), np.std([noisy_fid])))
print('noise-free state tomography fidelity = {:.4f} \u00B1 {:.4f}'.format(np.mean([noisefree_fid]), np.std([noisefree_fid])))
print('mitigated state tomography fidelity = {:.4f} \u00B1 {:.4f}'.format(np.mean([mitigated_fid]), np.std([mitigated_fid])))



unmitigated = np.mean([noisy_fid])
ideal = np.mean([noisefree_fid])
mitigated = np.mean([mitigated_fid])

error_unmitigated = abs(unmitigated-ideal)
error_mitigated = abs(mitigated-ideal)

print("Error (unmitigated):", error_unmitigated)
print("Error (mitigated):", error_mitigated)

print("Relative error (unmitigated):", (error_unmitigated/ideal))
print("Relative error (mitigatedR):", error_mitigated/ideal)

print(f"Error reduction: {(error_unmitigated-error_mitigated)/error_unmitigated :.1%}.")