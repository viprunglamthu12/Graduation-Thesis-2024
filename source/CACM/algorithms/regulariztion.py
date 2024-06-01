import numpy as np
import torch
from CACM.algorithms.utils import mmn_compute
    
class Regularizer:
    def __init__(
        self,
        E_conditioned,
        ci_test,
        kernel_type,
        gamma,         
    ):
        self.E_conditioned = E_conditioned
        self.ci_test = ci_test
        self.kernel_type = kernel_type
        self.gamma = gamma
    
    def mmd(self, x, y):
        return mmn_compute(x, y, self.kernel_type, self.gamma)
    
    def unconditional_reg(self, classifs, attribute_labels, num_envs, E_eq_A=False):
        penalty = 0

        if E_eq_A:
            if self.E_conditioned is False:
                for i in range(num_envs):
                    for j in range(i+1, num_envs):
                        penalty += self.mmd(classifs[i], classifs[j])
        
        else:
            if self.E_conditioned:
                for i in range(num_envs):
                    unique_attr_labels = torch.unique(attribute_labels[i])
                    unique_attr_labels_indices = []
                    for label in unique_attr_labels:
                        label_ind = [ind for ind, j in enumerate(attribute_labels[i]) if j == label]
                        unique_attr_labels_indices.append(label_ind)
                    
                    nulabels = unique_attr_labels.shape[0]
                    for aidx in range(nulabels):
                        for bidx in range(aidx + 1, nulabels):
                            penalty += self.mmd(
                                classifs[i][unique_attr_labels_indices[aidx]],
                                classifs[i][unique_attr_labels_indices[bidx]]
                            )
            
            else:
                overall_nmb_indices, nmb_id = [], []
                for i in range(num_envs):
                    unique_attrs = torch.unique(attribute_labels[i])
                    unique_attr_indices = []
                    for attr in unique_attrs:
                        attr_ind = [ind for ind, j in enumerate(attribute_labels[i]) if j == attr]
                        unique_attr_indices.append(attr_ind)
                        overall_nmb_indices.append(attr_ind)
                        nmb_id.append(i)
                
                nuattr = len(overall_nmb_indices)
                for aidx in range(nuattr):
                    for bidx in range(aidx + 1, nuattr):
                        a_nmb_id = nmb_id[aidx]
                        b_nmb_id = nmb_id[bidx]
                        penalty += self.mmd(
                            classifs[a_nmb_id][overall_nmb_indices[aidx]],
                            classifs[b_nmb_id][overall_nmb_indices[bidx]],
                        )
        return penalty
    
    def conditional_reg(self, classifs, attribute_labels, conditioning_subset, num_envs, E_eq_A=False):
        """
        Implement conditional regularization φ(x) ⊥⊥ A_i | A_s

        :param classifs: feature representations output from classifier layer (gφ(x))
        :param attribute_labels: attribute labels loaded with the dataset for attribute A_i
        :param conditioning_subset: list of subset of observed variables A_s (attributes + targets) such that (X_c, A_i) are d-separated conditioned on this subset
        :param num_envs: number of environments/domains
        :param E_eq_A: Binary flag indicating whether attribute (A_i) coinicides with environment (E) definition

        Find group indices for conditional regularization based on conditioning subset by taking all possible combinations
        e.g., conditioning_subset = [A1, Y], where A1 is in {0, 1} and Y is in {0, 1, 2},
        we assign groups in the following way:
            A1 = 0, Y = 0 -> group 0
            A1 = 1, Y = 0 -> group 1
            A1 = 0, Y = 1 -> group 2
            A1 = 1, Y = 1 -> group 3
            A1 = 0, Y = 2 -> group 4
            A1 = 1, Y = 2 -> group 5

        Code snippet for computing group indices adapted from WILDS: https://github.com/p-lambda/wilds
            @inproceedings{wilds2021,
             title = {{WILDS}: A Benchmark of in-the-Wild Distribution Shifts},
             author = {Pang Wei Koh and Shiori Sagawa and Henrik Marklund and Sang Michael Xie and Marvin Zhang and Akshay Balsubramani and Weihua Hu and Michihiro Yasunaga and Richard Lanas Phillips and Irena Gao and Tony Lee and Etienne David and Ian Stavness and Wei Guo and Berton A. Earnshaw and Imran S. Haque and Sara Beery and Jure Leskovec and Anshul Kundaje and Emma Pierson and Sergey Levine and Chelsea Finn and Percy Liang},
             booktitle = {International Conference on Machine Learning (ICML)},
             year = {2021}
            }`

        """

        penalty = 0

        if E_eq_A:  # Environment (E) and attribute (A) coincide
            if self.E_conditioned is False:  # there is no correlation between E and X_c
                overall_group_vindices = {}  # storing group indices
                overall_group_eindices = {}  # storing corresponding environment indices

                for i in range(num_envs):
                    conditioning_subset_i = [subset_var[i] for subset_var in conditioning_subset]
                    conditioning_subset_i_uniform = [
                        ele.unsqueeze(1) if ele.dim() == 1 else ele for ele in conditioning_subset_i
                    ]
                    grouping_data = torch.cat(conditioning_subset_i_uniform, 1)
                    assert grouping_data.min() >= 0, "Group numbers cannot be negative."
                    cardinality = 1 + torch.max(grouping_data, dim=0)[0]
                    cumprod = torch.cumprod(cardinality, dim=0)
                    n_groups = cumprod[-1].item()
                    if torch.cuda.is_available():
                        factors = torch.cat((torch.tensor([1], device='cuda'), cumprod[:-1].to('cuda')))
                        factors = factors.double()

                        group_indices = grouping_data.double().cuda() @ factors
                    else:
                        factors_np = np.concatenate(([1], cumprod[:-1]))
                        factors = torch.from_numpy(factors_np)

                        group_indices = grouping_data @ factors

                    for group_idx in range(n_groups):
                        group_idx_indices = [
                            gp_idx for gp_idx in range(len(group_indices)) if group_indices[gp_idx] == group_idx
                        ]

                        if group_idx not in overall_group_vindices:
                            overall_group_vindices[group_idx] = {}
                            overall_group_eindices[group_idx] = {}

                        unique_attrs = torch.unique(
                            attribute_labels[i][group_idx_indices]
                        )  # find distinct attributes in environment with same group_idx_indices
                        unique_attr_indices = []
                        for attr in unique_attrs:  # storing indices with same attribute value and group label
                            if attr not in overall_group_vindices[group_idx]:
                                overall_group_vindices[group_idx][attr] = []
                                overall_group_eindices[group_idx][attr] = []
                            single_attr = []
                            for group_idx_indices_attr in group_idx_indices:
                                if attribute_labels[i][group_idx_indices_attr] == attr:
                                    single_attr.append(group_idx_indices_attr)
                            overall_group_vindices[group_idx][attr].append(single_attr)
                            overall_group_eindices[group_idx][attr].append(i)
                            unique_attr_indices.append(single_attr)

                for (group_label) in (overall_group_vindices):  
                    # applying MMD penalty between distributions P(φ(x)|ai, g), P(φ(x)|aj, g) i.e samples with different attribute labelues but same group label
                    tensors_list = []
                    for attr in overall_group_vindices[group_label]:
                        attrs_list = []
                        if overall_group_vindices[group_label][attr] != []:
                            for il_ind, indices_list in enumerate(overall_group_vindices[group_label][attr]):
                                attrs_list.append(
                                    classifs[overall_group_eindices[group_label][attr][il_ind]][indices_list]
                                )
                        if len(attrs_list) > 0:
                            tensor_attrs = torch.cat(attrs_list, 0)
                            tensors_list.append(tensor_attrs)

                    nuattr = len(tensors_list)
                    for aidx in range(nuattr):
                        for bidx in range(aidx + 1, nuattr):
                            penalty += self.mmd(tensors_list[aidx], tensors_list[bidx])

        else:
            if self.E_conditioned:
                for i in range(num_envs):
                    conditioning_subset_i = [subset_var[i] for subset_var in conditioning_subset]
                    conditioning_subset_i_uniform = [
                        ele.unsqueeze(1) if ele.dim() == 1 else ele for ele in conditioning_subset_i
                    ]
                    grouping_data = torch.cat(conditioning_subset_i_uniform, 1)
                    assert grouping_data.min() >= 0, "Group numbers cannot be negative."
                    cardinality = 1 + torch.max(grouping_data, dim=0)[0]
                    cumprod = torch.cumprod(cardinality, dim=0)
                    n_groups = cumprod[-1].item()
                    if torch.cuda.is_available():
                        factors = torch.cat((torch.tensor([1], device='cuda'), cumprod[:-1].to('cuda')))
                        factors = factors.double()
                        
                        group_indices = grouping_data.double().cuda() @ factors
                    else:
                        factors_np = np.concatenate(([1], cumprod[:-1]))
                        factors = torch.from_numpy(factors_np)

                        group_indices = grouping_data @ factors

                    for group_idx in range(n_groups):
                        group_idx_indices = [
                            gp_idx for gp_idx in range(len(group_indices)) if group_indices[gp_idx] == group_idx
                        ]
                        unique_attrs = torch.unique(
                            attribute_labels[i][group_idx_indices]
                        )  # find distinct attributes in environment with same group_idx_indices
                        unique_attr_indices = []
                        for attr in unique_attrs:
                            single_attr = []
                            for group_idx_indices_attr in group_idx_indices:
                                if attribute_labels[i][group_idx_indices_attr] == attr:
                                    single_attr.append(group_idx_indices_attr)
                            unique_attr_indices.append(single_attr)

                        nuattr = unique_attrs.shape[0]
                        for aidx in range(nuattr):
                            for bidx in range(aidx + 1, nuattr):
                                penalty += self.mmd(
                                    classifs[i][unique_attr_indices[aidx]], classifs[i][unique_attr_indices[bidx]]
                                )

            else:
                overall_group_vindices = {}  # storing group indices
                overall_group_eindices = {}  # storing corresponding environment indices

                for i in range(num_envs):
                    conditioning_subset_i = [subset_var[i] for subset_var in conditioning_subset]
                    conditioning_subset_i_uniform = [
                        ele.unsqueeze(1) if ele.dim() == 1 else ele for ele in conditioning_subset_i
                    ]
                    grouping_data = torch.cat(conditioning_subset_i_uniform, 1)
                    assert grouping_data.min() >= 0, "Group numbers cannot be negative."
                    cardinality = 1 + torch.max(grouping_data, dim=0)[0]
                    cumprod = torch.cumprod(cardinality, dim=0)
                    n_groups = cumprod[-1].item()
                    if torch.cuda.is_available():
                        factors = torch.cat((torch.tensor([1], device='cuda'), cumprod[:-1].to('cuda')))
                        factors = factors.double()
                        
                        group_indices = grouping_data.double().cuda() @ factors
                    else:
                        factors_np = np.concatenate(([1], cumprod[:-1]))
                        factors = torch.from_numpy(factors_np)

                        group_indices = grouping_data @ factors

                    for group_idx in range(n_groups):
                        group_idx_indices = [
                            gp_idx for gp_idx in range(len(group_indices)) if group_indices[gp_idx] == group_idx
                        ]

                        if group_idx not in overall_group_vindices:
                            overall_group_vindices[group_idx] = {}
                            overall_group_eindices[group_idx] = {}

                        unique_attrs = torch.unique(
                            attribute_labels[i][group_idx_indices]
                        )  # find distinct attributes in environment with same group_idx_indices
                        unique_attr_indices = []
                        for attr in unique_attrs:  # storing indices with same attribute value and group label
                            if attr not in overall_group_vindices[group_idx]:
                                overall_group_vindices[group_idx][attr] = []
                                overall_group_eindices[group_idx][attr] = []
                            single_attr = []
                            for group_idx_indices_attr in group_idx_indices:
                                if attribute_labels[i][group_idx_indices_attr] == attr:
                                    single_attr.append(group_idx_indices_attr)
                            overall_group_vindices[group_idx][attr].append(single_attr)
                            overall_group_eindices[group_idx][attr].append(i)
                            unique_attr_indices.append(single_attr)

                for (
                    group_label
                ) in (
                    overall_group_vindices
                ):  # applying MMD penalty between distributions P(φ(x)|ai, g), P(φ(x)|aj, g) i.e samples with different attribute labelues but same group label
                    tensors_list = []
                    for attr in overall_group_vindices[group_label]:
                        attrs_list = []
                        if overall_group_vindices[group_label][attr] != []:
                            for il_ind, indices_list in enumerate(overall_group_vindices[group_label][attr]):
                                attrs_list.append(
                                    classifs[overall_group_eindices[group_label][attr][il_ind]][indices_list]
                                )
                        if len(attrs_list) > 0:
                            tensor_attrs = torch.cat(attrs_list, 0)
                            tensors_list.append(tensor_attrs)

                    nuattr = len(tensors_list)
                    for aidx in range(nuattr):
                        for bidx in range(aidx + 1, nuattr):
                            penalty += self.mmd(tensors_list[aidx], tensors_list[bidx])

        return penalty
