from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch
from torch import nn
import numpy as np
from tqdm import tqdm

class KBCModel(nn.Module, ABC):
    def get_ranking(self, queries: torch.Tensor, filters: Dict[Tuple[int, int], List[int]], batch_size: int = 1000, chunk_size: int = -1):
        ranks = torch.ones(len(queries))
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    scores, _ = self.forward(these_queries)
                    targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]  
                        scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum((scores >= targets).float(), dim=1).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks

class COCA(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
        super(COCA, self).__init__()
        self.sizes = sizes
        self.rank = rank
        alpha = 0.1
        gamma = 0.8
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=False)
        wn_ling_f = r"../pre_train/matrix_wn_ling.npy"
        wn_visual_f = r"../pre_train/matrix_wn_visual.npy"
        wn_ling,wn_visual = torch.tensor(np.load(wn_ling_f)),torch.tensor(np.load(wn_visual_f))        
        self.img_vec = wn_visual.to(torch.float32)
        self.img_dimension = wn_visual.shape[-1]
        self.ling_vec = wn_ling.to(torch.float32)
        self.ling_dimension = wn_ling.shape[-1]
        self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 4 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_img)
        self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 4 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_ling)
        self.embeddings = nn.ModuleList([nn.Embedding(s, 4 * rank, sparse=True) for s in sizes[:2]])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
           
    def forward(self, x):
        device = x.device
        img_embeddings = self.img_vec.to(device).mm(self.mats_img.to(device))
        ling_embeddings = self.ling_vec.to(device).mm(self.mats_ling.to(device))
        embedding = (1 - self.alpha-self.gamma) * self.embeddings[0].weight + self.alpha * img_embeddings + self.gamma * ling_embeddings
        lhs = embedding[(x[:, 0])]
        rel = self.embeddings[1](x[:, 1])
        rhs = embedding[(x[:, 2])]   
        lhs = lhs[:, :self.rank], lhs[:, self.rank:2*self.rank], lhs[:, 2*self.rank:3*self.rank], lhs[:, 3*self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:2*self.rank], rel[:, 2*self.rank:3*self.rank], rel[:, 3*self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:2*self.rank], rhs[:, 2*self.rank:3*self.rank], rhs[:, 3*self.rank:]
        to_score = embedding
        to_score = to_score[:, :self.rank], to_score[:, self.rank:2*self.rank], to_score[:, 2*self.rank:3*self.rank], to_score[:, 3*self.rank:]
        rel_r, rel_i, rel_j, rel_k = rel[0], rel[1], rel[2], rel[3]
        A = lhs[0] * rel_r - lhs[1] * rel_i - lhs[2] * rel_j - lhs[3] * rel_k
        B = lhs[0] * rel_i + rel_r * lhs[1] + lhs[2] * rel_k - rel_j * lhs[3]
        C = lhs[0] * rel_j + rel_r * lhs[2] + lhs[3] * rel_i - rel_k * lhs[1]
        D = lhs[0] * rel_k + rel_r * lhs[3] + lhs[1] * rel_j - rel_i * lhs[2]
        return (A @ to_score[0].transpose(0, 1) + B @ to_score[1].transpose(0, 1) + C @ to_score[2].transpose(0, 1) + D @ to_score[3].transpose(0, 1)), (
                    torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2 + lhs[2] ** 2 + lhs[3] ** 2),
                    torch.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2 + rel[3] ** 2),
                    torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2 + rhs[2] ** 2 + rhs[3] ** 2))