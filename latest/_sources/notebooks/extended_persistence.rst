.. code:: ipython3

    import numpy as np
    from scipy.sparse import coo_matrix
    
    from gtda.externals.python.ripser_interface import ripser

.. code:: ipython3

    def cone_over_graph(graph):
        n_points = max(graph.shape)
        n_diag = min(graph.shape)
        graph = graph.tocoo()
        data_diag = np.zeros(n_points + 1, dtype=graph.dtype)
        data_diag[:n_diag] = graph.diagonal()
        max_value, min_value = data_diag[:n_diag].max(), data_diag[:n_diag].min()
        data_diag[-1] = min_value - 1
        off_diag = graph.row != graph.col
        row_off_diag, col_off_diag = graph.row[off_diag], graph.col[off_diag]
        row = np.concatenate([row_off_diag,
                              np.arange(n_points)])
        col = np.concatenate([col_off_diag,
                              np.full(n_points, n_points)])
        data = np.concatenate([np.maximum(data_diag[row_off_diag],
                                          data_diag[col_off_diag]),
                               2*max_value + 1 - data_diag[:n_points]])
        graph = coo_matrix((data, (row, col)), shape=(n_points + 1, n_points + 1))
        graph.setdiag(data_diag)
        graph_struct=dict(graph=graph, max_value=max_value, min_value=min_value)
        return graph_struct
    
    def compute_ph(graph):
        dgms = ripser(graph, metric='precomputed')['dgms']
        return dgms
    
    def transform_output(dgms, max_value, min_value):
        for i in range(len(dgms)):
            mask_down_sweep = dgms[i] > max_value
            sgn = 2 * np.logical_not(np.logical_xor.reduce(mask_down_sweep, axis=1, keepdims=True)).astype(int) - 1
            dgms[i][mask_down_sweep] = 2 * max_value + 1 -  dgms[i][mask_down_sweep]
            if not i:
                dgms[i] = np.hstack([dgms[i][:-1, :], np.full((len(dgms[i]) - 1, 1), i), sgn[:-1, :]])
            else:
                dgms[i] = np.hstack([dgms[i], np.full((len(dgms[i]), 1), i), sgn])
        return dgms
        
    def extended_persistence(graph):
        new_graph_struct = cone_over_graph(graph)
        dgms = compute_ph(new_graph_struct["graph"])
        return transform_output(dgms, max_value=new_graph_struct["max_value"],
                                min_value=new_graph_struct["min_value"])

.. code:: ipython3

    graph = coo_matrix(
        (np.array([0, 1, 2, 3, 4, 5, 0.5, 4.5, -1, 1, 2, 3, 4, 4, 5, 2, 4.5]),
         (np.array([*range(9), *[0, 1, 1, 2, 3, 4, 2, 3]]),
          np.array([*range(9), *[1, 2, 3, 4, 4, 5, 6, 7]])))
    )

.. code:: ipython3

    extended_persistence(graph)




.. parsed-literal::

    [array([[ 0.5,  2. ,  0. ,  1. ],
            [ 0. ,  5. ,  0. , -1. ],
            [-1. , -1. ,  0. , -1. ]]),
     array([[ 4.5,  3. ,  1. ,  1. ],
            [ 4. ,  1. ,  1. , -1. ]])]







.. parsed-literal::

    [array([[ 0.5,  2. ],
            [ 0. ,  inf],
            [-1. ,  inf]]),
     array([[ 4., inf]])]



