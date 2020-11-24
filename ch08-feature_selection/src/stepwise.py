# -*- coding: utf-8 -*-
# 张春强
# 《机器学习：软件工程方法与实现》 第8章-特征选择代码

import numpy as np
import pandas as pd

class StepWise:
    def __init__(self, X, y, func_model=None):
        '''stepwise 回归变量选择，通用的方法
        使用p-value和BIC评价
        
        X:dataframe
        y:serise
        supprort
        ----------
            1:forward
            2:backward
            3:forward/backward
            4:backward/forward

        sle:select in
        sls:drop out
        func_model:带fit的回归建模函数
        n_features_to_select 为空，表示没有数量限制。
                优先级低于sle 和 sls，即只有在满足sle/sls的情况下该参数才起作用
        return
        -------
            选中的列
        usage
        -----
        st = StepWise(df_test[x_cols],df_test['label'])
        selected = st.fb(n_features_to_select=5,verbose=True,sle=0.2)

        '''
        self.func_model = func_model if func_model is not None else StepWise.sm_logit
        self.X = X
        self.y = y
   
    @staticmethod
    def sm_logit(X, y):# 高斯，柏松等自行添加
        return sm.GLM(
            y, sm.add_constant(X), family=sm.families.Binomial()).fit()

    def _print(self, pvs, bak_print, flag):
        t = 'select' if flag else 'remove'
        if pvs.notna().any():
            print('{}:{:30},BIC={:.6f},pv={:.6f}'.format(
                t, pvs.idxmin(), pvs.min(), bak_print[pvs.idxmin()]))     
        
        
    def _forward(self, X, y, selected, sle=0.1, verbose=False):
        ''' 得到候选特征中最好的一个变量，要求pv小于sle '''
        candidate = list(set(X.columns.tolist()) - set(selected))
        pvs = pd.Series(index=candidate)
        bak_print = {}
        for cc in candidate:
            m = self.func_model(X[selected + [cc]], y)
            # 加入bic是否为空的判断
            if m.pvalues[cc] < sle:
                if pd.isna(m.bic):  #导致na，表明该特征的加入反而不稳定
                    print('bic is na,set to np.inf')
                    pvs[cc] = np.inf
                else:
                    pvs[cc] = m.bic
                bak_print[cc] = m.pvalues[cc]
            elif verbose:
                print('  p-value of {:30} >={:.6f}(sle),exclude(donot select)'.
                      format(cc, m.pvalues[cc]))
    
        if len(bak_print) == 0:
            print('All p-values >={}(sle)'.format(sle))
            return None
        if verbose:
            self._print(pvs, bak_print, 1)
    
        if pvs.notna().any():
            return pvs.idxmin()
        else:
            print('All BIC ARE NaN!!')
            return None


    def forward(self, sle=0.1, n_features_to_select=None, verbose=False):
        selected = []
        # 注意n_features_to_select is None在前
        assert (n_features_to_select is None or n_features_to_select > 0
                ), 'n_features_to_select need larger than 0 or None'
        n = np.inf if n_features_to_select is None else n_features_to_select
    
        while True and len(selected) < n:
            best = self._forward(self.X,
                                 self.y,
                                 selected=selected,
                                 sle=sle,
                                 verbose=verbose)
            if best is not None:
                selected.append(best)
            else:
                break
        return selected
    
    
    def _backward(self, X, y, removed, sls=0.04, verbose=False):
        ''' 得到已有特征中最差的一个变量,要求pv大于sls '''
        candidate = list(set(X.columns.tolist()) - set(removed))
        if len(candidate) == 0: return None
    
        pvs = pd.Series(index=candidate)
        m = self.func_model(X[candidate], y)
        pvs = m.pvalues
        pvs = pvs[~(pvs.index == 'const')]
        pvs = pvs[pvs > sls]
    
        candidate = pvs.index.tolist()
        bak_print = pvs.copy()
    
        if len(candidate) == 0:
            if verbose:
                print('  All p-values <={}(sls),donot remove'.format(sls))
            return None
    
        selected = list(set(X.columns.tolist()) - set(candidate) - set(removed))
    
        pvs = pd.Series(index=candidate)
        for cc in candidate:
            t = (candidate + selected).copy()
            t.remove(cc)
            m = self.func_model(X[t], y)
            pvs[cc] = m.bic
    
        if verbose:
            self._print(pvs, bak_print, 0)
        # 这里潜在一个意思：如果去除某个cc后bic为Nan，也说明该cc整体起到的作用，不应该去除
        if pvs.notna().any():
            return pvs.idxmin()
        else:
            print('All BIC ARE NaN!!')
            return None


    def backward(self, sls=0.04, n_features_to_select=None, verbose=False):
        removed = []
        assert (n_features_to_select is None or n_features_to_select > 0
                ), 'n_features_to_select need larger than 0 or None'
        # 都取反
        n = np.inf if n_features_to_select is None else (self.X.shape[1] -
                                                         n_features_to_select)
    
        while True and len(removed) < n:
            worse = self._backward(self.X,
                                   self.y,
                                   removed=removed,
                                   sls=sls,
                                   verbose=verbose)
            if worse is not None:
                removed.append(worse)
            else:
                break
        return [cc for cc in self.X.columns.tolist() if cc not in removed]
    
    def fb(self,
       sle=0.1,
       sls=0.04,
       n_features_to_select=None,
       verbose=False,
       n_threshold=3):
        selected = []
        assert (n_features_to_select is None or n_features_to_select > 0
                ), 'n_features_to_select need larger than 0 or None'
        n = np.inf if n_features_to_select is None else n_features_to_select
    
        while True and len(selected) < n:
            best = self._forward(self.X,
                                 self.y,
                                 selected=selected,
                                 sle=sle,
                                 verbose=verbose)
            if best is not None:
                selected.append(best)
            else:
                break
    
            if len(selected) < n_threshold:
                continue
            worse = self._backward(self.X[selected],
                                   self.y,
                                   removed=[],
                                   sls=sls,
                                   verbose=verbose)
            if worse is not None:
                if selected[-1] == worse:
                    print(
                        'remove threshold:{},select threshold:{},infinite loop!!!'.
                        format(sls, sle))
                    break
                selected.remove(worse)
    
        return selected
    
    
    def bf(self,
       sls=0.04,
       sle=0.1,
       n_features_to_select=None,
       verbose=False,
       n_threshold=3):

        removed = []
        assert (n_features_to_select is None or n_features_to_select > 0
                ), 'n_features_to_select need larger than 0 or None'
        # 都取反
        n = np.inf if n_features_to_select is None else (self.X.shape[1] -
                                                         n_features_to_select)
    
        while True and len(removed) < n:
            worse = self._backward(self.X,
                                   self.y,
                                   removed=removed,
                                   sls=sls,
                                   verbose=verbose)
            if worse is not None:
                removed.append(worse)
            else:
                break
    
            if len(removed) < n_threshold:
                continue
            selected = list(set(self.X.columns.tolist()) - set(removed))
            best = self._forward(self.X,
                                 self.y,
                                 selected=selected,
                                 sle=sle,
                                 verbose=verbose)
            if best is not None:
                if removed[-1] == best:
                    if verbose:
                        print(
                            'remove threshold:{},select threshold:{},infinite loop!!!'
                            .format(sls, sle))
                    break
            removed.remove(best)
        return [cc for cc in self.X.columns.tolist() if cc not in removed]
        