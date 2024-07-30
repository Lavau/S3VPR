import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
import faiss.contrib.torch_utils
from prettytable import PrettyTable 


def test_val_sets(args, model, dataloaders):
    val_set_names = args.val_set_names
    recalls = {}
    for val_set_name, dataloader in zip(val_set_names, dataloaders):
        recalls[val_set_name] = test_val_set(args, model, val_set_name, dataloader, False) 
    return recalls


def test_val_set(args, model, val_set_name, dataloader, print_results=True):
    val_dataset = dataloader.dataset
    model = model.eval()

    with torch.no_grad():
        all_features = [] 
        for inputs, _ in tqdm(dataloader, desc=f"validation {val_set_name} iteration", ncols=100): 
            features = model(inputs.to(args.device))
            features = features.cpu()
            all_features.append(features)
    
    all_features = torch.concat(all_features, dim=0) 

    if 'pitts' in val_set_name:
        num_references = val_dataset.dbStruct.numDb
        positives = val_dataset.getPositives()
    elif 'msls' in val_set_name:
        num_references = val_dataset.num_references
        positives = val_dataset.pIdx
    elif 'nordland' in val_set_name:
        num_references = val_dataset.num_references
        positives = val_dataset.ground_truth
    elif 'tokyo247' in val_set_name:
        num_references = val_dataset.num_references
        positives = val_dataset.pIdx
    elif 'SPED' in val_set_name:
        num_references = val_dataset.num_references
        positives = val_dataset.ground_truth
    else:
        raise Exception(f'Please implement validation_epoch_end for {val_set_name}') 
    
    """this return descriptors in their order depending on how the validation dataset is implemented 
       for this project (MSLS val, Pittburg val), it is always references then queries
       [R1, R2, ..., Rn, Q1, Q2, ...]
    """
    database_features = all_features[ : num_references]
    queries_features = all_features[num_references : ]

    k_values = [1, 5, 10, 15, 20, 50, 100]
    pitts_dict, predictions = get_validation_recalls(
        r_list=database_features, 
        q_list=queries_features,
        k_values=k_values,
        gt=positives,
        print_results=print_results,
        dataset_name=val_set_name
    )
    del database_features, queries_features, all_features 
 
    recalls_str =", ".join([f"R@{k}: {pitts_dict[k]:.4f}" for k in k_values])
    logging.info(f'Ranking results: {val_set_name} {recalls_str}')  
    return pitts_dict


def get_validation_recalls(r_list, q_list, k_values, gt, print_results=True, faiss_gpu=False, dataset_name='dataset without name ?', testing=False):
        embed_size = r_list.shape[1]
        if faiss_gpu:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
        # build index
        else:
            faiss_index = faiss.IndexFlatL2(embed_size)

        if r_list.dtype != torch.float32:
            r_list = r_list.type(torch.float32)
        if q_list.dtype != torch.float32:
            q_list = q_list.type(torch.float32)
                    
        # add references
        faiss_index.add(r_list)

        # search for queries in the index
        _, predictions = faiss_index.search(q_list, max(k_values))
        
        if testing:
            return predictions
        
        # start calculating recall_at_k
        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], gt[q_idx])):
                    correct_at_k[i:] += 1
                    break
        
        correct_at_k = correct_at_k / len(predictions)
        d = {k:v for (k,v) in zip(k_values, correct_at_k)}

        if print_results:
            print() # print a new line
            table = PrettyTable()
            table.field_names = ['K']+[str(k) for k in k_values]
            table.add_row(['Recall@K']+ [f'{v:.4f}' for v in correct_at_k])
            print(table.get_string(title=f"First rank performances on {dataset_name}"))
        
        return d, predictions
 