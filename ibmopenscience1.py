from qiskit import assemble, execute, transpile, Aer
from collections import Counter
from functools import reduce
from collections import Counter
from functools import reduce
from qiskit.opflow import Zero, One
from qiskit.ignis.verification.tomography import StateTomographyFitter
from qiskit.quantum_info import state_fidelity


shots = 1024



t_qc = transpile(qc, sim_noisy_jakarta)
qobj = assemble(t_qc)
counts = sim_noisy_jakarta.run(qobj, shots=shots).result().get_counts()

t_qc_sim = transpile(qc, sim)
noiseless_result = sim.run(assemble(t_qc_sim), shots=shots).result()
noiseless_counts = noiseless_result.get_counts()
    
print("noisy:      ", counts)
print("noise-free: ", noiseless_counts)





def sorted_counts(counts):
    complete = dict(reduce(lambda a, b: a.update(b) or a, [{'000': 0, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 0}, counts], Counter()))
    return {k: v for k, v in sorted(complete.items(), key=lambda item: item[0])}


zipped = list(zip(sorted_counts(noiseless_counts).values(), sorted_counts(counts).values()))
modifier = list(map(lambda pair: pair[0]/pair[1], zipped))

print("modifier: ", modifier)



print("noise-free: ", sorted_counts(noiseless_counts))

counts = sorted_counts(sim_noisy_jakarta.run(qobj, shots=shots).result().get_counts())
print("noisy:      ", counts)

mitigated = {item[0]: item[1]*modifier[i] for i, item in enumerate(counts.items())}
print("mitigated:  ", mitigated)





st_qcs = state_tomography_circuits(qc, [1,3,5])
print ("There are {} circuits in the list".format(len(st_qcs)))

def get_modifiers(qc):
    
    t_qc_sim = transpile(qc, sim)
    noiseless_result = sim.run( assemble(t_qc_sim), shots=shots).result()
    noiseless_counts = sorted_counts(noiseless_result.get_counts())
    
    t_qc = transpile(qc, sim_noisy_jakarta)
    qobj = assemble(t_qc)
    counts = sorted_counts(sim_noisy_jakarta.run(qobj, shots=shots).result().get_counts())
    
    zipped = list(zip(noiseless_counts.values(), counts.values()))
    modifier = list(map(lambda pair: pair[0]/pair[1], zipped))

    print("noisy:     ", counts)
    print("nose-free: ", noiseless_counts)
    
    print("modifier: ", modifier)
    print("\n")
    
    return modifier


modifiers = list(map(get_modifiers, st_qcs))





# Compute the state tomography based on the st_qcs quantum circuits and the results from those ciricuits
def state_tomo(result, st_qcs, mitigate=False):
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


noisy_job = execute(st_qcs, sim_noisy_jakarta, shots=shots)
noisefree_job = execute(st_qcs, sim, shots=shots)
    
noisy_fid = state_tomo(noisy_job.result(), st_qcs, mitigate=False)
noisefree_fid = state_tomo(noisefree_job.result(), st_qcs, mitigate=False)
mitigated_fid = state_tomo(noisefree_job.result(), st_qcs, mitigate=True)

print('noisy state tomography fidelity = {:.4f} \u00B1 {:.4f}'.format(np.mean([noisy_fid]), np.std([noisy_fid])))
print('noise-free state tomography fidelity = {:.4f} \u00B1 {:.4f}'.format(np.mean([noisefree_fid]), np.std([noisefree_fid])))
print('mitigated state tomography fidelity = {:.4f} \u00B1 {:.4f}'.format(np.mean([mitigated_fid]), np.std([mitigated_fid])))

