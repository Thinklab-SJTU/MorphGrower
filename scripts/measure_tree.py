import json
import os
from utils.utils import load_neuron, load_neurons
import numpy as np
from tqdm import tqdm

def measure(output_dir, data_dir, split_json, is_morph=False):
    ############### log init ###############
    if output_dir != '' and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ############### data load ###############
    neurons, neuron_file = load_neurons(data_dir, scaling=1.0, return_filelist=True)
    
    records = []
    Compartment_number_num = 0
    N_stems_sum = 0
    N_branch_sum = 0
    N_bifs_sum = 0
    Branch_order_sum = 0
    Contraction_sum = 0
    Length_sum = 0
    Branch_pathlength_sum = 0
    PathDistance_sum = 0
    EucDistance_sum = 0
    Bif_ampl_local_sum = 0
    Bif_ampl_remote_sum = 0
    pc_angle_remote_sum = 0
    pc_angle_local_sum = 0
    sum_EucDistan = 0
    cnt=0

    if split_json!=None:
        log = json.load(open(split_json))
        test_split = log["data_split"]["test"]
        reidx = log["reidx"]

    for neuron, file in tqdm(zip(neurons,neuron_file), total=len(neuron_file)):
        if split_json!=None:
            if file in reidx and not (reidx[file] in test_split):
                continue
        cnt += 1
        #print(file)
        record = {'idx': file,
            'Compartment number': neuron.CompartmentNumber(),
            'N_stems':neuron.N_stems(),
            'N_branch':neuron.N_branch(),
            'N_bifs':neuron.N_bifs(),
            'Branch_order':neuron.Branch_order(),
            'Contraction':neuron.Contraction(),
            'Length':neuron.Length(),
            'Branch_pathlength':neuron.Branch_pathlength(),
            "PathDistance":neuron.PathDistance(),
            "EucDistance":neuron.EucDistance(),
            "Bif_ampl_local":neuron.Bif_ampl_local() if not is_morph else [0,0,0],
            "Bif_ampl_remote":neuron.Bif_ampl_remote() if not is_morph else [0,0,0],
            "pc_angle_remote":neuron.pc_angle_remote() if not is_morph else [0,0,0],
            "pc_angle_local":neuron.pc_angle_local() if not is_morph else [0,0,0],
        }
        Compartment_number_num += record['Compartment number']
        N_stems_sum += record['N_stems'][0]
        N_branch_sum += record['N_branch'][0]
        N_bifs_sum += record['N_bifs'][0]
        Branch_order_sum += record['Branch_order'][3]
        Contraction_sum += record['Contraction'][2]
        Length_sum += record['Length'][2]
        Branch_pathlength_sum += record['Branch_pathlength'][2]
        PathDistance_sum += record['PathDistance'][3]
        EucDistance_sum += record['EucDistance'][3]
        Bif_ampl_local_sum += record['Bif_ampl_local'][2]
        Bif_ampl_remote_sum += record['Bif_ampl_remote'][2]
        sum_EucDistan += record['EucDistance'][0]
        pc_angle_remote_sum += record['pc_angle_remote'][2]
        pc_angle_local_sum += record['pc_angle_local'][2]
        records.append(record)
    #print(cnt)
    l = cnt
    mean_record = {
            'Compartment number': Compartment_number_num/l,
            'N_stems':N_stems_sum/l,
            'N_branch':N_branch_sum/l,
            'N_bifs':N_bifs_sum/l,
            'Branch_order':Branch_order_sum/l,
            'Contraction':Contraction_sum/l,
            'Length':Length_sum/l,
            'Branch_pathlength':Branch_pathlength_sum/l,
            "PathDistance":PathDistance_sum/l,
            "EucDistance":EucDistance_sum/l,
            "Bif_ampl_local":Bif_ampl_local_sum/l,
            "Bif_ampl_remote":Bif_ampl_remote_sum/l,
            "sum_EucDistan": sum_EucDistan/l,
            "pc_angle_remote":pc_angle_remote_sum/l,
            "pc_angle_local":pc_angle_local_sum/l,
        }
    #print(mean_record)
    final_records = {'data_dir':data_dir, 'average':mean_record, 'records': records}
    with open(os.path.join(output_dir,'result.json'), 'w') as Fout:
        json.dump(final_records, Fout, indent=4)
    return [mean_record['N_stems'],
            mean_record['N_branch'],
            mean_record['N_bifs'],
            mean_record['Branch_order'],
            mean_record['Contraction'],
            mean_record['Length'],
            mean_record['Branch_pathlength'],
            mean_record["PathDistance"],
            mean_record["EucDistance"],
            mean_record["Bif_ampl_local"],
            mean_record["Bif_ampl_remote"],
            mean_record["sum_EucDistan"],
            mean_record["pc_angle_remote"],
            mean_record["pc_angle_local"],
           ]

def measure_forest(output_dir, data_dir, split_json):
    ############### log init ###############
    if output_dir != '' and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ############### data load ###############
    neurons, neuron_file = load_neurons(data_dir, scaling=1.0, return_filelist=True)

    records = []
    Compartment_number_num = 0
    N_stems_sum = 0
    N_branch_sum = []
    N_bifs_sum = []
    Branch_order_sum = []
    Contraction_sum = []
    Length_sum = []
    Branch_pathlength_sum = []
    PathDistance_sum = []
    EucDistance_sum = []
    Bif_ampl_local_sum = []
    Bif_ampl_remote_sum = []
    pc_angle_remote_sum = []
    pc_angle_local_sum = []

    cnt=0

    if split_json!=None:
        log = json.load(open(split_json))
        test_split = log["data_split"]["test"]
        reidx = log["reidx"]

    for neuron, file in zip(neurons,neuron_file):
        if split_json!=None:
            if file in reidx and  not (reidx[file] in test_split):
                continue
        cnt += 1
        record = {'idx': file,
            'Compartment number': neuron.CompartmentNumber(),
            'N_stems':neuron.N_stems(),
            'N_branch':neuron.N_branch_forest(),
            'N_bifs':neuron.N_bifs_forest(),
            'Branch_order':neuron.Branch_order_forest(),
            'Contraction':neuron.Contraction_forest(),
            'Length':neuron.Length_forest(),
            'Branch_pathlength':neuron.Branch_pathlength_forest(),
            "PathDistance":neuron.PathDistance_forest(),
            "EucDistance":neuron.EucDistance_forest(),
            #"Bif_ampl_local":neuron.Bif_ampl_local_forest(),
            #"Bif_ampl_remote":neuron.Bif_ampl_remote_forest(),
            "pc_angle_remote":neuron.pc_angle_forest_remote(),
            "pc_angle_local":neuron.pc_angle_forest_local(),
        }
        Compartment_number_num += record['Compartment number']
        N_stems_sum += record['N_stems'][0]
        N_branch_sum.extend(record['N_branch'])
        N_bifs_sum.extend(record['N_bifs'])
        Branch_order_sum.extend(record['Branch_order'])
        Contraction_sum.extend(record['Contraction'])
        Length_sum.extend(record['Length'])
        Branch_pathlength_sum.extend(record['Branch_pathlength'])
        PathDistance_sum.extend(record['PathDistance'])
        EucDistance_sum.extend(record['EucDistance'])
        #Bif_ampl_local_sum.extend(record['Bif_ampl_local'])
        #Bif_ampl_remote_sum.extend(record['Bif_ampl_remote'])
        pc_angle_remote_sum.extend(record['pc_angle_remote'])
        pc_angle_local_sum.extend(record['pc_angle_local'])
        records.append(record)

    N_branch_sum = np.array([x[0] for x in N_branch_sum])
    N_bifs_sum = np.array([x[0] for x in N_bifs_sum])
    Branch_order_sum = np.array([x[3] for x in Branch_order_sum])
    Contraction_sum = np.array([x[2] for x in Contraction_sum])
    Length_sum = np.array([x[2] for x in Length_sum])
    Branch_pathlength_sum = np.array([x[2] for x in Branch_pathlength_sum])
    PathDistance_sum = np.array([x[3] for x in PathDistance_sum])
    sum_EucDistan = np.array([x[0] for x in EucDistance_sum])
    EucDistance_sum = np.array([x[3] for x in EucDistance_sum])
    #Bif_ampl_local_sum = np.array([x[3] for x in Bif_ampl_local_sum])
    #Bif_ampl_remote_sum = np.array([x[3] for x in Bif_ampl_remote_sum])
    pc_angle_remote_sum = np.array([x[2] for x in pc_angle_remote_sum])
    pc_angle_local_sum = np.array([x[2] for x in pc_angle_local_sum])
    #print(cnt)
    l = cnt
    mean_record = {
            'Compartment number': Compartment_number_num/l,
            'N_stems':N_stems_sum/l,
            'N_branch':np.mean(N_branch_sum),
            'N_bifs':np.mean(N_bifs_sum),
            'Branch_order':np.mean(Branch_order_sum),
            'Contraction':np.mean(Contraction_sum),
            'Length':np.mean(Length_sum),
            'Branch_pathlength':np.mean(Branch_pathlength_sum),
            "PathDistance":np.mean(PathDistance_sum),
            "EucDistance":np.mean(EucDistance_sum),
            #"Bif_ampl_local":np.mean(Bif_ampl_local_sum),
            #"Bif_ampl_remote":np.mean(Bif_ampl_remote_sum),
            "sum_EucDistan":np.mean(sum_EucDistan),
            "pc_angle_remote":np.mean(pc_angle_remote_sum),
            "pc_angle_local":np.mean(pc_angle_local_sum),
        }
    final_records = {'data_dir':data_dir, 'average':mean_record, 'records': records}
    with open(os.path.join(output_dir,'result.json'), 'w') as Fout:
        json.dump(final_records, Fout, indent=4)
    return [mean_record['N_stems'],
            mean_record['N_branch'],
            mean_record['N_bifs'],
            mean_record['Branch_order'],
            mean_record['Contraction'],
            mean_record['Length'],
            mean_record['Branch_pathlength'],
            mean_record["PathDistance"],
            mean_record["EucDistance"],
            #mean_record["Bif_ampl_local"],
            #mean_record["Bif_ampl_remote"],
            mean_record["sum_EucDistan"],
            mean_record["pc_angle_remote"],
            mean_record["pc_angle_local"],]

def run(info, measure_name="measure", is_morph=False):
    result = []
    output_dirs,data_dirs,split_jsons = info["output_dir"], info["data_dir"], info["split_json"]
    measure_fun = measure if measure_name=="measure" else measure_forest
    for output_dir,data_dir,split_json in zip(output_dirs,data_dirs,split_jsons):
        result.append(measure_fun(output_dir, data_dir, split_json, is_morph=False))
    result = np.array(result).transpose()
    std = np.std(result[:,-5:],-1)
    avg = np.mean(result[:,-5:],-1)

    print("\n".join([ '\t'.join(map(str, line))+'\t'+str(a)+'\t'+str(s) for line, a, s in zip(result,avg,std)]))

