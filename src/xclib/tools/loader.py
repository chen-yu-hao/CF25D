from pathlib import Path
import os
from xclib.tools.pyscf_io import *
def load_names(chk_dir):
    """在 `chk_dir` 下收集待加载的 pychk 文件路径。

    仅处理文件名包含 ``"pychk"`` 的条目；以 (目录, 前缀, 扩展名) 分组，其中前缀为 stem
    去掉可选的 ``-RKS`` 后缀后的部分。若同一组同时存在普通版本与 ``-RKS`` 版本，则优先保留
    ``-RKS`` 版本。

    Parameters
    ----------
    chk_dir : str
        checkpoint 文件所在目录路径（当前实现会直接与文件名拼接，建议以 `/` 结尾）。

    Returns
    -------
    list[str]
        选中的文件路径列表（字符串）。
    """
    # chk_dir = "../checkfiles/NC15_checkfile/"
    list_dir = os.listdir(chk_dir)
    pick = {}
    for f in list_dir:
        if "pychk" in f:
            p = Path(f)
            stem = p.stem
            is_rks = stem.endswith("-RKS")
            prefix = stem[:-4] if is_rks else stem
            key = (p.parent, prefix, p.suffix)  # 同目录+同前缀+同扩展名算一组

            if is_rks:
                pick[key] = f              # 有 -RKS 就覆盖为 -RKS
            else:
                pick.setdefault(key, f)    # 没有 -RKS 才保留原始版本

    kept_files = list(pick.values())
    return [chk_dir+i for i in kept_files]
# import jax
# jax.config.update('jax_enable_x64', True)
import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Tuple
# import jax.numpy as np
import time
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Callable, Any
import time
from scipy.optimize import minimize

from functools import partial
from copy import deepcopy
def reshape_pad_weight(x, block=10_000):
    x = np.asarray(x)               # (2, 6, N)
    N = x.shape[-1]
    n = (N + block - 1) // block     # ceil(N / 10000)
    pad = n * block - N
    x = np.pad(x, [(0,0)]*(x.ndim-1) + [(0, pad)])  # 末尾补 0
    return x.reshape(*x.shape[:-1], n, block)         # (2, 6, n, 10000)
def reshape_pad_rho(x, block=10_000):
    x = np.asarray(x)               # (2, 6, N)
    N = x.shape[-1]
    n = (N + block - 1) // block     # ceil(N / 10000)
    pad = n * block - N
    x = np.pad(x, [(0,0)]*(x.ndim-1) + [(0, pad)], constant_values=1)  # 末尾补 0
    return x.reshape(*x.shape[:-1], n, block)         # (2, 6, n, 10000)
class baseKKData():
    def __init__(self, ) -> None:
        self.values: Any
        self.key_map = {}
        self.offset = 0

    def insert(self, k1, k2, vector) -> None:
        if k1 not in self.key_map:
            self.key_map[k1] = {}
        if k2 not in self.key_map[k1]:
            if self.offset >= self.values.shape[0]:
                raise Exception("Error> There is no space in Array.")
            self.key_map[k1][k2] = self.offset
            self.offset += 1
            self.values[self.key_map[k1][k2]] = vector
    def __getitem__(self, k1_k2) -> None:
        k1, k2 = k1_k2
        if k1 in self.key_map and k2 in self.key_map[k1]:
            row_idx = self.key_map[k1][k2]
            return self.values[row_idx]
        else:
            raise Exception("Error> k1: ({}), k2: ({}) not found.".format(k1, k2))

class KKMatrix(baseKKData):
    def __init__(self,  shape: Tuple[int, int]):
        super().__init__()
        self.values: NDArray = np.zeros(shape = shape, dtype=float)

class KKKFunc(baseKKData):
    def __init__(self):
        super().__init__()
        self.values: List = []
        
    def insert(self, k1, k2, k3: str, operator: Callable) -> None:
        if k1 not in self.key_map:
            self.key_map[k1] = {}
        if k2 not in self.key_map[k1]:
            self.key_map[k1][k2] = {}
        if k3 not in self.key_map[k1][k2]:
            # list 是可变的取消了越界检测, 抽象base还是不太好
            self.key_map[k1][k2][k3] = self.offset
            self.offset += 1
            self.values.append(operator)
    def __getitem__(self, k1_k2_k3) -> None:
        k1, k2, k3 = k1_k2_k3
        if k1 in self.key_map and k2 in self.key_map[k1] and k3 in self.key_map[k1][k2]:
            row_idx = self.key_map[k1][k2][k3]
            return self.values[row_idx]
        else:
            raise Exception("Error> k1: ({}), k2: ({}), k3: ({}) not found.".format(k1, k2, k3))
# class loss_results():
#     def __init(self, loss):
#         for 
class CFXXD_Train_Data_Set():
    def __init__(self, database: Dict, ref_root_path:str, fit_root_path: str, cut: list, weight: Dict):
        self.weight_dict = weight
        self.database = database
        self.rrp = ref_root_path
        self.frp = fit_root_path
        self.cut = cut
        self.ref_data_raw = self.__init_ref_data__()
        self.fit_data_raw = self.__init_fit_data__()
        self.init_fit_data_sort(fit_root_path)
        self.array_arg()
        fit_data, ratios, masks = self.array_from_dict(cut)
        self.fit_data = fit_data
        self.ratios = ratios
        self.masks = masks
    def init_fit_data_sort(self, workpath: str)-> Dict: 
        arg=0
        self.fit_data_Ene_raw_sort = {}
        for k,v in self.database.items():
            fit_data_fp = "{}/{}".format(workpath, k)
            Name_s, value_s = self.read_fit_file(fit_data_fp)
            self.fit_data_Ene_raw_sort[k] = {}
            for i, fit_name in enumerate(Name_s):
                self.fit_data_Ene_raw_sort[k][fit_name] = arg
                arg+=1
        # return fit_data_Ene_raw_sort
    def rho_loader(self,workpath,level = 0):
        self.weights_list = []
        self.rho_list = []
        for k,v in self.database.items():
            rho_data_fp = "{}/{}".format(workpath, k.replace('_ref','')+"_checkfile")
            fit_data_fp = "{}/{}".format(self.frp, k)
            Name_s, value_s = self.read_fit_file(fit_data_fp)
            for i, fit_name in enumerate(Name_s):
                try:
                    self.weights_list.append(load_mf_rks(rho_data_fp+"/"+fit_name+"-RKS.pychk",level=level).grids.weights)
                    self.rho_list.append(np.load(rho_data_fp+"/"+fit_name+"-RKS.npy"))
                except:
                    self.weights_list.append(load_mf_uks(rho_data_fp+"/"+fit_name+".pychk",level=level).grids.weights)
                    self.rho_list.append(np.load(rho_data_fp+"/"+fit_name+".npy"))
            
    def array_from_dict(self, cut=[3,7,42]):
        ref_data_Ene = self.ref_data_raw
        fit_data_Ene_raw = self.fit_data_raw
        count=[0,0,0]
        for k, v in ref_data_Ene.items():
            # mol_ene_result = fit_data_Ene_sort[k]
            
            for ref, job_bag in v.items():
                for job in job_bag:
                    if len(job["ratio_s"])<=3:
                        count[0]+=1
                    elif len(job["ratio_s"])<=7:
                        count[1]+=1
                    else:
                        count[2]+=1
                        # print(len(job["ratio_s"]))
        total = np.sum(count)
        fit_data = []
        ratios = []
        masks = []
        # has_ratioo = []
        for i,j in zip(count,cut):
            fit_data.append(np.zeros([i,j,*fit_data_Ene_raw.shape[1:]]))
            ratios.append(np.zeros([i,j+3]))
            masks.append(np.zeros([i,j], dtype=bool))
        count=[0,0,0]
        self.weight = [[],[],[]]
        self.weight1 = [[],[],[]]
        weight_counter = 0
        for k, v in ref_data_Ene.items():
            for ref, job_bag in v.items():
                # print(len(job_bag))
                for job in job_bag:
                    names = job["name_s"]
                    if len(job["ratio_s"])<=cut[0]:
                        fit_data[0][count[0],:len(job["ratio_s"])]  +=fit_data_Ene_raw[names][0]
                        masks[0][count[0],:len(job["ratio_s"])] = True
                        ratios[0][count[0],:len(job["ratio_s"])]+=job["ratio_s"]
                        ratios[0][count[0],-3] +=job["has_ratioo"]
                        ratios[0][count[0],-2] +=job["ratioo"]
                        ratios[0][count[0],-1] +=job["ref_s"][0]
                        count[0]+=1
                        self.weight[0].append(1/self.weight_dict[ref])
                        self.weight1[0].append(1/self.weight_dict[ref]/len(job_bag))
                    elif len(job["ratio_s"])<=cut[1]:
                        fit_data[1][count[1],:len(job["ratio_s"])]  +=fit_data_Ene_raw[names][0]
                        masks[1][count[1],:len(job["ratio_s"])] = True
                        ratios[1][count[1],:len(job["ratio_s"])]+=job["ratio_s"]
                        ratios[1][count[1],-3] +=job["has_ratioo"]
                        ratios[1][count[1],-2] +=job["ratioo"]
                        ratios[1][count[1],-1] +=job["ref_s"][0]
                        count[1]+=1
                        self.weight[1].append(1/self.weight_dict[ref])
                        self.weight1[1].append(1/self.weight_dict[ref]/len(job_bag))
                    else:
                        fit_data[2][count[2],:len(job["ratio_s"])]  +=fit_data_Ene_raw[names][0]
                        masks[2][count[2],:len(job["ratio_s"])] = True
                        ratios[2][count[2],:len(job["ratio_s"])]+=job["ratio_s"]
                        ratios[2][count[2],-3] +=job["has_ratioo"]
                        ratios[2][count[2],-2] +=job["ratioo"]
                        ratios[2][count[2],-1] +=job["ref_s"][0]
                        count[2]+=1
                        self.weight[2].append(1/self.weight_dict[ref])
                        self.weight1[2].append(1/self.weight_dict[ref]/len(job_bag))
        self.weight[0]=np.array(self.weight[0])
        self.weight[1]=np.array(self.weight[1])
        self.weight[2]=np.array(self.weight[2])
        self.weight1[0]=np.array(self.weight1[0])
        self.weight1[1]=np.array(self.weight1[1])
        self.weight1[2]=np.array(self.weight1[2])
        return fit_data, ratios, masks
    def array_from_dict_rho(self, cut=[3,7,42]):
        ref_data_Ene = self.ref_data_raw
        fit_data_Ene_raw = self.fit_data_raw
        count=[0,0,0]
        for k, v in ref_data_Ene.items():
            # mol_ene_result = fit_data_Ene_sort[k]
            
            for ref, job_bag in v.items():
                for job in job_bag:
                    if len(job["ratio_s"])<=3:
                        count[0]+=1
                    elif len(job["ratio_s"])<=7:
                        count[1]+=1
                    else:
                        count[2]+=1
                        # print(len(job["ratio_s"]))
        total = np.sum(count)
        fit_data = []
        ratios = []
        masks = []
        # has_ratioo = []
        for i,j in zip(count,cut):
            fit_data.append(np.zeros([i,j,*fit_data_Ene_raw.shape[1:]]))
            ratios.append(np.zeros([i,j+3]))
            masks.append(np.zeros([i,j], dtype=bool))
        count=[0,0,0]
        self.weight = [[],[],[]]
        self.weight1 = [[],[],[]]
        self.rhos = [[], [], []]
        self.weights = [[], [], []]
        weight_counter = 0
        for k, v in ref_data_Ene.items():
            for ref, job_bag in v.items():
                # print(len(job_bag))
                for job in job_bag:
                    names = job["name_s"]
                    if len(job["ratio_s"])<=cut[0]:
                        # print(names)
                        fit_data[0][count[0],:len(job["ratio_s"])]  +=fit_data_Ene_raw[names][0]
                        self.weights[0].append([self.weights_list[name] for name in names[0]])
                        self.rhos[0].append([self.rho_list[name] for name in names[0]])
                        if len(names[0])!=cut[0]:
                            for i in range(cut[0]-len(names[0])):
                                self.weights[0][-1].append(np.zeros([1]))
                                self.rhos[0][-1].append(np.ones([2,6,1]))
                            
                        masks[0][count[0],:len(job["ratio_s"])] = True
                        ratios[0][count[0],:len(job["ratio_s"])]+=job["ratio_s"]
                        ratios[0][count[0],-3] +=job["has_ratioo"]
                        ratios[0][count[0],-2] +=job["ratioo"]
                        ratios[0][count[0],-1] +=job["ref_s"][0]
                        count[0]+=1
                        self.weight[0].append(1/self.weight_dict[ref])
                        self.weight1[0].append(1/self.weight_dict[ref]/len(job_bag))
                    elif len(job["ratio_s"])<=cut[1]:
                        fit_data[1][count[1],:len(job["ratio_s"])]  +=fit_data_Ene_raw[names][0]
                        self.weights[1].append([self.weights_list[name] for name in names[0]])
                        self.rhos[1].append([self.rho_list[name] for name in names[0]])
                        if len(names[0])!=cut[1]:
                            for i in range(cut[1]-len(names[0])):
                                self.weights[1][-1].append(np.zeros([1]))
                                self.rhos[1][-1].append(np.ones([2,6,1]))
                        masks[1][count[1],:len(job["ratio_s"])] = True
                        ratios[1][count[1],:len(job["ratio_s"])]+=job["ratio_s"]
                        ratios[1][count[1],-3] +=job["has_ratioo"]
                        ratios[1][count[1],-2] +=job["ratioo"]
                        ratios[1][count[1],-1] +=job["ref_s"][0]
                        count[1]+=1
                        self.weight[1].append(1/self.weight_dict[ref])
                        self.weight1[1].append(1/self.weight_dict[ref]/len(job_bag))
                    else:
                        self.weights[2].append([self.weights_list[name] for name in names[0]])
                        self.rhos[2].append([self.rho_list[name] for name in names[0]])
                        if len(names[0])!=cut[2]:
                            for i in range(cut[2]-len(names[0])):
                                self.weights[2][-1].append(np.zeros([1]))
                                self.rhos[2][-1].append(np.ones([2,6,1]))
                        masks[2][count[2],:len(job["ratio_s"])] = True
                        ratios[2][count[2],:len(job["ratio_s"])]+=job["ratio_s"]
                        ratios[2][count[2],-3] +=job["has_ratioo"]
                        ratios[2][count[2],-2] +=job["ratioo"]
                        ratios[2][count[2],-1] +=job["ref_s"][0]
                        count[2]+=1
                        self.weight[2].append(1/self.weight_dict[ref])
                        self.weight1[2].append(1/self.weight_dict[ref]/len(job_bag))
        self.weight[0]=np.array(self.weight[0])
        self.weight[1]=np.array(self.weight[1])
        self.weight[2]=np.array(self.weight[2])
        self.weight1[0]=np.array(self.weight1[0])
        self.weight1[1]=np.array(self.weight1[1])
        self.weight1[2]=np.array(self.weight1[2])
        return fit_data, ratios, masks
    def array_arg(self):
        MUE = []
     
        for k, v in self.ref_data_raw.items():
            mol_ene_result = self.fit_data_Ene_raw_sort[k]
            for ref, job_bag in v.items():
                # print(ref)
                for job in job_bag:
                    names = job["name"]
                    # print(names)
                    job["name_s"] =np.array([[mol_ene_result[n] for n in names]])
    def __init_ref_data__(self):
        ref_data_Ene = {}
        for k,v in self.database.items():
            ref_data_Ene[k] = {}
            for f in v:
                ref_data_fp_s = "{}/{}".format(self.rrp, f)
                # 如果遇到异常的数据点直接不读取
                ref_bag = self.__read_ref_file__(ref_data_fp_s, Ignore_error=True)
                ref_data_Ene[k][f] = ref_bag
        return ref_data_Ene
    def read_fit_file(self,fp: str) -> tuple[list[str], NDArray]:
        Ename = []
        
        i = 0
        with open(fp, 'r') as F:
            lines= F.readlines()
            values = np.zeros([len(lines),78])
            for line in lines:
                line = line.split()
                try:
                    values[i] += np.array([float(i) for i in line[1:]])
                except:
                    print(np.array([float(i) for i in line[1:]]).shape)
                    print(line)
                Ename.append(line[0])
                i+=1
        return Ename, values
    def __init_fit_data__(self) -> Dict: 
        fit_data_Ene_raw = []
        for k,v in self.database.items():
            # print(f"reading {k}")
            fit_data_fp = "{}/{}".format(self.frp, k)
            Name_s, value_s = self.read_fit_file(fit_data_fp)
            # fit_data_Ene_raw[k] = []
            for i, fit_name in enumerate(Name_s):
                # print(value_s)
                fit_data_Ene_raw.append(value_s[i])
        return np.array(fit_data_Ene_raw)
    def __read_ref_file__(self, fp: str, Ignore_error: bool = True) -> List:
        result = []
        support_n_col = [3, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 25, 43, 85, 13, 17]

        with open(fp, 'r') as F:
            lines = F.readlines()
            for line_num, line in enumerate(lines):
                out = {}
                line = line.split()
                n_col = len(line)
                out["job_type"] = n_col
                out["name"] = []
                out["ratio_s"] = []
                out["ref_s"] = []
                out["ratioo"] = 1.0
                out["has_ratioo"] = 627.509               # 此处增加该标志，防止极端情况下文件中读取的ratioo为627.509
                if n_col % 2 == 0 and n_col in support_n_col:    
                    out["name"].extend([line[i] for i in range(1, n_col-1, 2)])
                    out["ratio_s"].extend([np.float64(line[i]) for i in range(0, n_col-2, 2)])
                    out["ref_s"].append(np.float64(line[n_col-2]))
                    out["ratioo"] = np.float64(line[n_col-1])
                    out["has_ratioo"] = 1.0
                elif n_col % 2 != 0 and n_col in support_n_col:
                    out["name"].extend([line[i] for i in range(1, n_col-1, 2)])
                    out["ratio_s"].extend([np.float64(line[i]) for i in range(0, n_col-2, 2)])
                    out["ref_s"].append(np.float64(line[n_col-1]))
                else:
                    if Ignore_error:
                        print("Not support n_col = {}, line_num = {}, content is {}".format(n_col, line_num, line))
                        continue
                    else:
                        raise Exception("Not support n_col = {}, line_num = {}, content is {}".format(n_col, line_num, line))
                out["ref_s"] = np.array(out["ref_s"])
                out["ratio_s"] = np.array(out["ratio_s"])
                result.append(out)
        return result

    def __update_fit_data__(self) -> KKMatrix:
        n_vector = 0
        dim_vector = None
        for k1, v1 in self.fit_data_raw.items():
            for k2, v2 in v1.items():
                if dim_vector == None:
                    dim_vector = v2.shape[0]
                n_vector += 1

        fit_data_KKMatrix = KKMatrix(shape=(n_vector, dim_vector))
        for k1, v1 in self.fit_data_raw.items():
            for k2, v2 in v1.items():
                fit_data_KKMatrix.insert(k1, k2, v2)

        return fit_data_KKMatrix

    def __loss__(self, k1, bag, fit_data) -> Tuple[float, float]:

        names = bag["name_s"]
        ratios = bag["ratio_s"]
        ref_ene = bag["ref_s"]
        ratioo = bag["ratioo"]
        has_ratioo = bag["has_ratioo"] 

        _pred_ene = np.dot(np.array([fit_data[k1, _] for _ in names]), np.array(ratios))

        if has_ratioo:      
            # 偶数
            _pred_ene = _pred_ene*627.509
            mue = np.abs(_pred_ene - ref_ene)/ratioo
        else:
            # 奇数
            mue = np.abs(_pred_ene - ref_ene) * 627.509

        return mue
    
    def __update_ref_data__(self, __loss__:Callable):
        ref_data_KKKFunc = KKKFunc()
        for k1, v1 in self.ref_data_raw.items():
            for k2, bag in v1.items():
                for i, i_bag in enumerate(bag):
                    k3 = str(i)
                    func = partial(__loss__, k1, i_bag)
                    ref_data_KKKFunc.insert(k1, k2,k3, func)
        return ref_data_KKKFunc
    
    def __call__(self):
        fit_data_KKMatrix = self.__update_fit_data__()
        ref_data_KKKFunc = self.__update_ref_data__(self.__loss__)
        return fit_data_KKMatrix, ref_data_KKKFunc
    def dict_from_array(self,loss):
        cut = self.cut
        ref_data_Ene = self.ref_data_raw
        fit_data_Ene_raw = self.fit_data_raw
        count=[0,0,0]
        total = [len(i) for i in self.masks]
        loss_results = {}
        for k, v in ref_data_Ene.items():
            loss_results[k] = {}
            loss_result = loss_results[k]
            for ref, job_bag in v.items():
                loss_result[ref]=[]
                for job in job_bag:
                    if len(job["ratio_s"])<=3:
                        loss_result[ref].append(loss[count[0]]/self.weight1[0][count[0]])
                        count[0]+=1
                    elif len(job["ratio_s"])<=7:
                        loss_result[ref].append(loss[count[1]+total[0]]/self.weight1[1][count[1]])
                        count[1]+=1
                    else:
                        loss_result[ref].append(loss[count[2]+total[0]+total[1]]/self.weight1[2][count[2]])
                        count[2]+=1
                        # print(len(job["ratio_s"]))
                loss_result[ref] = np.array(loss_result[ref])
                loss_result[ref] = {"Array":loss_result[ref],"MUE":np.mean(np.abs(loss_result[ref])),"RMSE":np.sqrt(np.mean(loss_result[ref]**2))}
        return loss_results
    def dictsort(self):
        self.weight_sort = []
        self.ref_names = []
        cut = self.cut
        ref_data_Ene = self.ref_data_raw
        fit_data_Ene_raw = self.fit_data_raw
        count=[0,0,0]
        total = [len(i) for i in self.masks]
        loss_results = []
        for k, v in ref_data_Ene.items():
            for ref, job_bag in v.items():
                loss_result=[]
                for job in job_bag:
                    if len(job["ratio_s"])<=3:
                        loss_result.append(count[0])
                        count[0]+=1
                    elif len(job["ratio_s"])<=7:
                        loss_result.append(count[1]+total[0])
                        count[1]+=1
                    else:
                        loss_result.append(count[2]+total[0]+total[1])
                        count[2]+=1
                loss_result = np.array(loss_result)
                loss_results.append(loss_result)
                self.weight_sort.append(self.weight_dict[ref])
                self.ref_names.append(ref)
        self.weight_sort = np.array(self.weight_sort)
        return loss_results,self.weight_sort
        
