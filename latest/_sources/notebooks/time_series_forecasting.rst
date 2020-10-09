Topology in time series forecasting
===================================

This notebook shows how ``giotto-tda`` can be used to create topological
features for time series forecasting tasks, and how to integrate them
into ``scikit-learn``–compatible pipelines.

In particular, we will concentrate on topological features which are
created from consecutive **sliding windows** over the data. In sliding
window models, a single time series array ``X`` of shape
``(n_timestamps, n_features)`` is turned into a time series of windows
over the data, with a new shape
``(n_windows, n_samples_per_window, n_features)``. There are two main
issues that arise when building forecasting models with sliding windows:

1. ``n_windows`` is smaller than ``n_timestamps``. This is because we
   cannot have more windows than there are timestamps without padding
   ``X``, and this is not done by ``giotto-tda``.
   ``n_timestamps - n_windows`` is even larger if we decide to pick a
   large stride between consecutive windows.
2. The target variable ``y`` needs to be properly “aligned” with each
   window so that the forecasting problem is meaningful and e.g. we
   don’t “leak” information from the future. In particular, ``y`` needs
   to be “resampled” so that it too has length ``n_windows``.

To deal with these issues, ``giotto-tda`` provides a selection of
transformers with ``resample``, ``transform_resample`` and
``fit_transform_resample`` methods. These are inherited from a
``TransformerResamplerMixin`` base class. Furthermore, ``giotto-tda``
provides a drop-in replacement for ``scikit-learn``\ ’s ``Pipeline``
which extends it to allow chaining ``TransformerResamplerMixin``\ s with
regular ``scikit-learn`` estimators.

If you are looking at a static version of this notebook and would like
to run its contents, head over to
`GitHub <https://github.com/giotto-ai/giotto-tda/blob/master/examples/time_series_forecasting.ipynb>`__
and download the source.

See also
--------

-  `Topology of time
   series <https://giotto-ai.github.io/gtda-docs/latest/notebooks/time_series_classification.html>`__
   which explains how transforming time series into point clouds via the
   *Takens embedding* procedure makes topological feature extraction
   possible.
-  `Topological feature extraction using VietorisRipsPersistence and
   PersistenceEntropy <https://giotto-ai.github.io/gtda-docs/latest/notebooks/vietoris_rips_quickstart.html>`__
   for a quick introduction to general topological feature extraction in
   ``giotto-tda``.

**License: AGPLv3**

``SlidingWindow``
-----------------

Let us start with a simple example of a “time series” ``X`` with a
corresponding target ``y`` of the same length.

.. code:: ipython3

    import numpy as np
    
    n_timestamps = 10
    X, y = np.arange(n_timestamps), np.arange(n_timestamps) - n_timestamps
    X, y




.. parsed-literal::

    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
     array([-10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1]))



We can instantiate our sliding window transformer-resampler and run it
on the pair ``(X, y)``:

.. code:: ipython3

    from gtda.time_series import SlidingWindow
    
    window_size = 3
    stride = 2
    
    SW = SlidingWindow(size=window_size, stride=stride)
    X_sw, yr = SW.fit_transform_resample(X, y)
    X_sw, yr




.. parsed-literal::

    (array([[1, 2, 3],
            [3, 4, 5],
            [5, 6, 7],
            [7, 8, 9]]),
     array([-7, -5, -3, -1]))



We note a couple of things: - ``fit_transform_resample`` returns a pair:
the window-transformed ``X`` and the resampled and aligned ``y``. -
``SlidingWindow`` has made a choice for us on how to resample ``y`` and
line it up with the windows from ``X``: a window on ``X`` corresponds to
the *last* value in a corresponding window over ``y``. This is common in
time series forecasting where, for example, ``y`` could be a shift of
``X`` by one timestamp. - Some of the initial values of ``X`` may not be
found in ``X_sw``. This is because ``SlidingWindow`` only ensures the
*last* value is represented in the last window, regardless of the
stride.

Multivariate time series example: Sliding window + topology ``Pipeline``
------------------------------------------------------------------------

``giotto-tda``\ ’s topology transformers expect 3D input. But our
``X_sw`` above is 2D. How do we capture interesting properties of the
topology of input time series then? For univariate time series, it turns
out that a good way is to use the “time delay embedding” or “Takens
embedding” technique explained in the first part of `Topology of time
series <https://github.com/giotto-ai/giotto-tda/blob/master/examples/time_series_classification.ipynb>`__.
But as this involves an extra layer of complexity, we leave it for later
and concentrate for now on an example with a simpler API which also
demonstrates the use of a ``giotto-tda`` ``Pipeline``.

Surprisingly, this involves multivariate time series input!

.. code:: ipython3

    rng = np.random.default_rng(42)
    
    n_features = 2
    
    X = rng.integers(0, high=20, size=(n_timestamps, n_features), dtype=int)
    X




.. parsed-literal::

    array([[ 1, 15],
           [13,  8],
           [ 8, 17],
           [ 1, 13],
           [ 4,  1],
           [10, 19],
           [14, 15],
           [14, 15],
           [10,  2],
           [16,  9]])



We are interpreting this input as a time series in two variables, of
length ``n_timestamps``. The target variable is the same ``y`` as
before.

.. code:: ipython3

    SW = SlidingWindow(size=window_size, stride=stride)
    X_sw, yr = SW.fit_transform_resample(X, y)
    X_sw, yr




.. parsed-literal::

    (array([[[13,  8],
             [ 8, 17],
             [ 1, 13]],
     
            [[ 1, 13],
             [ 4,  1],
             [10, 19]],
     
            [[10, 19],
             [14, 15],
             [14, 15]],
     
            [[14, 15],
             [10,  2],
             [16,  9]]]),
     array([-7, -5, -3, -1]))



``X_sw`` is now a complicated-looking array, but it has a simple
interpretation. Again, ``X_sw[i]`` is the ``i``-th window on ``X``, and
it contains ``window_size`` samples from the original time series. This
time, the samples are not scalars but 1D arrays.

What if we suspect that the way in which the **correlations** between
the variables evolve over time can help forecast the target ``y``? This
is a common situation in neuroscience, where each variable could be data
from a single EEG sensor, for instance.

``giotto-tda`` exposes a ``PearsonDissimilarity`` transformer which
creates a 2D dissimilarity matrix from each window in ``X_sw``, and
stacks them together into a single 3D object. This is the correct format
(and information content!) for a typical topological transformer in
``gtda.homology``. See also `Topological feature extraction from
graphs <https://github.com/giotto-ai/giotto-tda/blob/master/examples/persistent_homology_graphs.ipynb>`__
for an in-depth look. Finally, we can extract simple scalar features
using a selection of transformers in ``gtda.diagrams``.

.. code:: ipython3

    from gtda.time_series import PearsonDissimilarity
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import Amplitude
    
    PD = PearsonDissimilarity()
    X_pd = PD.fit_transform(X_sw)
    VR = VietorisRipsPersistence(metric="precomputed")
    X_vr = VR.fit_transform(X_pd)  # "precomputed" required on dissimilarity data
    Ampl = Amplitude()
    X_a = Ampl.fit_transform(X_vr)
    X_a




.. parsed-literal::

    array([[0.18228669, 0.        ],
           [0.03606068, 0.        ],
           [0.28866041, 0.        ],
           [0.01781238, 0.        ]])



Notice that we are not acting on ``y`` above. We are simply creating
features from each window using topology! *Note*: it’s two features per
window because we used the default value for ``homology_dimensions`` in
``VietorisRipsPersistence``, not because we had two variables in the
time series initially!

We can now put this all together into a ``giotto-tda`` ``Pipeline``
which combines both the sliding window transformation on ``X`` and
resampling of ``y`` with the feature extraction from the windows on
``X``.

*Note*: while we could import the ``Pipeline`` class and use its
constructor, we use the convenience function ``make_pipeline`` instead,
which is a drop-in replacement for
`scikit-learn’s <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html>`__.

.. code:: ipython3

    from sklearn import set_config
    set_config(display='diagram')  # For HTML representations of pipelines
    
    from gtda.pipeline import make_pipeline
    
    pipe = make_pipeline(SW, PD, VR, Ampl)
    pipe




.. raw:: html

    <style>div.sk-top-container {color: black;background-color: white;}div.sk-toggleable {background-color: white;}label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}div.sk-estimator:hover {background-color: #d4ebff;}div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}div.sk-item {z-index: 1;}div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}div.sk-parallel-item:only-child::after {width: 0;}div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}div.sk-label-container {position: relative;z-index: 2;text-align: center;}div.sk-container {display: inline-block;position: relative;}</style><div class="sk-top-container"><div class="sk-container"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="507c0e71-9447-40d8-bef9-7ad46cc2fc00" type="checkbox" ><label class="sk-toggleable__label" for="507c0e71-9447-40d8-bef9-7ad46cc2fc00">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[('slidingwindow', SlidingWindow(size=3, stride=2)),
                    ('pearsondissimilarity', PearsonDissimilarity()),
                    ('vietorisripspersistence',
                     VietorisRipsPersistence(metric='precomputed')),
                    ('amplitude', Amplitude())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="a3cec379-edd4-42da-b555-732cd236f717" type="checkbox" ><label class="sk-toggleable__label" for="a3cec379-edd4-42da-b555-732cd236f717">SlidingWindow</label><div class="sk-toggleable__content"><pre>SlidingWindow(size=3, stride=2)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="adc9e789-64ca-444a-9938-2a118e2d49ba" type="checkbox" ><label class="sk-toggleable__label" for="adc9e789-64ca-444a-9938-2a118e2d49ba">PearsonDissimilarity</label><div class="sk-toggleable__content"><pre>PearsonDissimilarity()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="30a9ff73-02c0-4d01-9eb5-0451bc680d7d" type="checkbox" ><label class="sk-toggleable__label" for="30a9ff73-02c0-4d01-9eb5-0451bc680d7d">VietorisRipsPersistence</label><div class="sk-toggleable__content"><pre>VietorisRipsPersistence(metric='precomputed')</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="a29734b4-aada-43ec-a233-288da7ffeb2e" type="checkbox" ><label class="sk-toggleable__label" for="a29734b4-aada-43ec-a233-288da7ffeb2e">Amplitude</label><div class="sk-toggleable__content"><pre>Amplitude()</pre></div></div></div></div></div></div></div>



Finally, if we have a *regression* task on ``y`` we can add a final
estimator such as scikit-learn’s ``RandomForestRegressor`` as a final
step in the previous pipeline, and fit it!

.. code:: ipython3

    from sklearn.ensemble import RandomForestRegressor
    
    RFR = RandomForestRegressor()
    
    pipe = make_pipeline(SW, PD, VR, Ampl, RFR)
    pipe




.. raw:: html

    <style>div.sk-top-container {color: black;background-color: white;}div.sk-toggleable {background-color: white;}label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}div.sk-estimator:hover {background-color: #d4ebff;}div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}div.sk-item {z-index: 1;}div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}div.sk-parallel-item:only-child::after {width: 0;}div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}div.sk-label-container {position: relative;z-index: 2;text-align: center;}div.sk-container {display: inline-block;position: relative;}</style><div class="sk-top-container"><div class="sk-container"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="7f8fcfd1-4172-400d-932b-2da4ffe90a39" type="checkbox" ><label class="sk-toggleable__label" for="7f8fcfd1-4172-400d-932b-2da4ffe90a39">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[('slidingwindow', SlidingWindow(size=3, stride=2)),
                    ('pearsondissimilarity', PearsonDissimilarity()),
                    ('vietorisripspersistence',
                     VietorisRipsPersistence(metric='precomputed')),
                    ('amplitude', Amplitude()),
                    ('randomforestregressor', RandomForestRegressor())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="680b3c03-56f7-4196-9cd7-b7a13d1228ef" type="checkbox" ><label class="sk-toggleable__label" for="680b3c03-56f7-4196-9cd7-b7a13d1228ef">SlidingWindow</label><div class="sk-toggleable__content"><pre>SlidingWindow(size=3, stride=2)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="c4094706-a5ad-4b1f-ae0d-d0e897fd52ab" type="checkbox" ><label class="sk-toggleable__label" for="c4094706-a5ad-4b1f-ae0d-d0e897fd52ab">PearsonDissimilarity</label><div class="sk-toggleable__content"><pre>PearsonDissimilarity()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="fb01ecc8-7ebd-4db0-a92d-c156ce0775d9" type="checkbox" ><label class="sk-toggleable__label" for="fb01ecc8-7ebd-4db0-a92d-c156ce0775d9">VietorisRipsPersistence</label><div class="sk-toggleable__content"><pre>VietorisRipsPersistence(metric='precomputed')</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="42bd8f42-1afe-4372-a625-2fdb8b9bf105" type="checkbox" ><label class="sk-toggleable__label" for="42bd8f42-1afe-4372-a625-2fdb8b9bf105">Amplitude</label><div class="sk-toggleable__content"><pre>Amplitude()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="234c2054-dfb1-417f-9df6-02032dc4a540" type="checkbox" ><label class="sk-toggleable__label" for="234c2054-dfb1-417f-9df6-02032dc4a540">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor()</pre></div></div></div></div></div></div></div>



.. code:: ipython3

    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    score = pipe.score(X, y)
    y_pred, score




.. parsed-literal::

    (array([-5.96, -4.4 , -4.16, -2.4 ]), 0.76264)



Univariate time series – ``TakensEmbedding`` and ``SingleTakensEmbedding``
--------------------------------------------------------------------------

The first part of `Topology of time
series <https://github.com/giotto-ai/giotto-tda/blob/master/examples/time_series_classification.ipynb>`__
explains a commonly used technique for converting a univariate time
series into a single **point cloud**. Since topological features can be
extracted from any point cloud, this is a gateway to time series
analysis using topology. The second part of that notebook shows how to
transform a *batch* of time series into a batch of point clouds, and how
to extract topological descriptors from each of them independently.
While in that notebook this is applied to a time series classification
task, in this notebook we are concerned with topology-powered
*forecasting* from a single time series.

Reasoning by analogy with the multivariate case above, we can look at
sliding windows over ``X`` as small time series in their own right and
track the evolution of *their* topology against the variable of interest
(or against itself, if we are interested in unsupervised tasks such as
anomaly detection).

There are two ways in which we can implement this idea in
``giotto-tda``: 1. We can first apply a ``SlidingWindow``, and then an
instance of ``TakensEmbedding``. 2. We can *first* compute a global
Takens embedding of the time series via ``SingleTakensEmbedding``, which
takes us from 1D/column data to 2D data, and *then* partition the 2D
data of vectors into sliding windows via ``SlidingWindow``.

The first route ensures that we can run our “topological feature
extraction track” in parallel with other feature-generation pipelines
from sliding windows, without experiencing shape mismatches. The second
route seems a little upside-down and it is not generally recommended,
but it has the advantange that globally “optimal” parameters for the
“time delay” and “embedding dimension” parameters can be computed
automatically by ``SingleTakensEmbedding``.

Below is what each route would look like.

*Remark:* In the presence of noise, a small sliding window size is
likely to reduce the reliability of the estimate of the time series’
local topology.

Option 1: ``SlidingWindow`` + ``TakensEmbedding``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TakensEmbedding`` is not a ``TransformerResamplerMixin``, but this is
not a problem in the context of a ``Pipeline`` when we order things in
this way.

.. code:: ipython3

    from gtda.time_series import TakensEmbedding
    
    X = np.arange(n_timestamps)
    
    window_size = 5
    stride = 2
    
    SW = SlidingWindow(size=window_size, stride=stride)
    X_sw, yr = SW.fit_transform_resample(X, y)
    X_sw, yr




.. parsed-literal::

    (array([[1, 2, 3, 4, 5],
            [3, 4, 5, 6, 7],
            [5, 6, 7, 8, 9]]),
     array([-5, -3, -1]))



.. code:: ipython3

    time_delay = 1
    dimension = 2
    
    TE = TakensEmbedding(time_delay=time_delay, dimension=dimension)
    X_te = TE.fit_transform(X_sw)
    X_te




.. parsed-literal::

    array([[[1, 2],
            [2, 3],
            [3, 4],
            [4, 5]],
    
           [[3, 4],
            [4, 5],
            [5, 6],
            [6, 7]],
    
           [[5, 6],
            [6, 7],
            [7, 8],
            [8, 9]]])



.. code:: ipython3

    VR = VietorisRipsPersistence()  # No "precomputed" for point clouds
    Ampl = Amplitude()
    RFR = RandomForestRegressor()
    
    pipe = make_pipeline(SW, TE, VR, Ampl, RFR)
    pipe




.. raw:: html

    <style>div.sk-top-container {color: black;background-color: white;}div.sk-toggleable {background-color: white;}label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}div.sk-estimator:hover {background-color: #d4ebff;}div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}div.sk-item {z-index: 1;}div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}div.sk-parallel-item:only-child::after {width: 0;}div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}div.sk-label-container {position: relative;z-index: 2;text-align: center;}div.sk-container {display: inline-block;position: relative;}</style><div class="sk-top-container"><div class="sk-container"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="8947f0a9-385c-4c7f-9c1f-27b1cc4782cf" type="checkbox" ><label class="sk-toggleable__label" for="8947f0a9-385c-4c7f-9c1f-27b1cc4782cf">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[('slidingwindow', SlidingWindow(size=5, stride=2)),
                    ('takensembedding', TakensEmbedding()),
                    ('vietorisripspersistence', VietorisRipsPersistence()),
                    ('amplitude', Amplitude()),
                    ('randomforestregressor', RandomForestRegressor())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="843e99ec-4158-443e-8ca8-18e669bba051" type="checkbox" ><label class="sk-toggleable__label" for="843e99ec-4158-443e-8ca8-18e669bba051">SlidingWindow</label><div class="sk-toggleable__content"><pre>SlidingWindow(size=5, stride=2)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="068fadba-b237-456e-8a27-7bd6a09010ec" type="checkbox" ><label class="sk-toggleable__label" for="068fadba-b237-456e-8a27-7bd6a09010ec">TakensEmbedding</label><div class="sk-toggleable__content"><pre>TakensEmbedding()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="fdf0ff24-980e-4103-9382-09537be55a81" type="checkbox" ><label class="sk-toggleable__label" for="fdf0ff24-980e-4103-9382-09537be55a81">VietorisRipsPersistence</label><div class="sk-toggleable__content"><pre>VietorisRipsPersistence()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="03acb9c9-d5bd-4d68-b248-74bc14f1b826" type="checkbox" ><label class="sk-toggleable__label" for="03acb9c9-d5bd-4d68-b248-74bc14f1b826">Amplitude</label><div class="sk-toggleable__content"><pre>Amplitude()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="2f938c9c-a79f-41d2-b3b8-5280877b374d" type="checkbox" ><label class="sk-toggleable__label" for="2f938c9c-a79f-41d2-b3b8-5280877b374d">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor()</pre></div></div></div></div></div></div></div>



.. code:: ipython3

    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    score = pipe.score(X, y)
    y_pred, score




.. parsed-literal::

    (array([-3., -3., -3.]), 0.0)



Option 2: ``SingleTakensEmbeding`` + ``SlidingWindow``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that ``SingleTakensEmbedding`` is also a
``TransformerResamplerMixin``, and that the logic for
resampling/aligning ``y`` is the same as in ``SlidingWindow``.

.. code:: ipython3

    from gtda.time_series import SingleTakensEmbedding
    
    X = np.arange(n_timestamps)
    
    STE = SingleTakensEmbedding(parameters_type="search", time_delay=2, dimension=3)
    X_ste, yr = STE.fit_transform_resample(X, y)
    X_ste, yr




.. parsed-literal::

    (array([[0, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
            [6, 8],
            [7, 9]]),
     array([-8, -7, -6, -5, -4, -3, -2, -1]))



.. code:: ipython3

    window_size = 5
    stride = 2
    
    SW = SlidingWindow(size=window_size, stride=stride)
    X_sw, yr = SW.fit_transform_resample(X_ste, yr)
    X_sw, yr




.. parsed-literal::

    (array([[[1, 3],
             [2, 4],
             [3, 5],
             [4, 6],
             [5, 7]],
     
            [[3, 5],
             [4, 6],
             [5, 7],
             [6, 8],
             [7, 9]]]),
     array([-3, -1]))



From here on, it is easy to push a very similar pipeline through as in
the multivariate case:

.. code:: ipython3

    VR = VietorisRipsPersistence()  # No "precomputed" for point clouds
    Ampl = Amplitude()
    RFR = RandomForestRegressor()
    
    pipe = make_pipeline(STE, SW, VR, Ampl, RFR)
    pipe




.. raw:: html

    <style>div.sk-top-container {color: black;background-color: white;}div.sk-toggleable {background-color: white;}label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}div.sk-estimator:hover {background-color: #d4ebff;}div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}div.sk-item {z-index: 1;}div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}div.sk-parallel-item:only-child::after {width: 0;}div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}div.sk-label-container {position: relative;z-index: 2;text-align: center;}div.sk-container {display: inline-block;position: relative;}</style><div class="sk-top-container"><div class="sk-container"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="512f1651-f091-4a98-9b78-9f27988d7eda" type="checkbox" ><label class="sk-toggleable__label" for="512f1651-f091-4a98-9b78-9f27988d7eda">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[('singletakensembedding',
                     SingleTakensEmbedding(dimension=3, time_delay=2)),
                    ('slidingwindow', SlidingWindow(size=5, stride=2)),
                    ('vietorisripspersistence', VietorisRipsPersistence()),
                    ('amplitude', Amplitude()),
                    ('randomforestregressor', RandomForestRegressor())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="f5e1179c-de87-4459-be3a-72608b7420aa" type="checkbox" ><label class="sk-toggleable__label" for="f5e1179c-de87-4459-be3a-72608b7420aa">SingleTakensEmbedding</label><div class="sk-toggleable__content"><pre>SingleTakensEmbedding(dimension=3, time_delay=2)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="2b929c30-f346-4e17-abcb-a95f777c1be6" type="checkbox" ><label class="sk-toggleable__label" for="2b929c30-f346-4e17-abcb-a95f777c1be6">SlidingWindow</label><div class="sk-toggleable__content"><pre>SlidingWindow(size=5, stride=2)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="10afa915-7d5f-4e1e-948b-96364c689e2e" type="checkbox" ><label class="sk-toggleable__label" for="10afa915-7d5f-4e1e-948b-96364c689e2e">VietorisRipsPersistence</label><div class="sk-toggleable__content"><pre>VietorisRipsPersistence()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="2642f356-7a81-4292-b562-3b5c598e8329" type="checkbox" ><label class="sk-toggleable__label" for="2642f356-7a81-4292-b562-3b5c598e8329">Amplitude</label><div class="sk-toggleable__content"><pre>Amplitude()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="cb428af2-fee6-4fa9-8c01-f87d9a9ee2f0" type="checkbox" ><label class="sk-toggleable__label" for="cb428af2-fee6-4fa9-8c01-f87d9a9ee2f0">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor()</pre></div></div></div></div></div></div></div>



.. code:: ipython3

    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    score = pipe.score(X, y)
    y_pred, score




.. parsed-literal::

    (array([-2.07, -2.07]), -0.004899999999999904)



Integrating non-topological features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The best results are obtained when topological methods are used not in
isolation but in **combination** with other methods. Here’s an example
where, in parallel with the topological feature extraction from local
sliding windows using **Option 2** above, we also compute the mean and
variance in each sliding window. A ``scikit-learn`` ``FeatureUnion`` is
used to combine these very different sets of features into a single
pipeline object.

.. code:: ipython3

    from functools import partial
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.pipeline import FeatureUnion
    from sklearn.base import clone
    
    mean = FunctionTransformer(partial(np.mean, axis=1, keepdims=True))
    var = FunctionTransformer(partial(np.var, axis=1, keepdims=True))
    
    pipe_topology = make_pipeline(TE, VR, Ampl)
    
    feature_union = FeatureUnion([("window_mean", mean),
                                  ("window_variance", var),
                                  ("window_topology", pipe_topology)])
        
    pipe = make_pipeline(SW, feature_union, RFR)
    pipe




.. raw:: html

    <style>div.sk-top-container {color: black;background-color: white;}div.sk-toggleable {background-color: white;}label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}div.sk-estimator:hover {background-color: #d4ebff;}div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}div.sk-item {z-index: 1;}div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}div.sk-parallel-item:only-child::after {width: 0;}div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}div.sk-label-container {position: relative;z-index: 2;text-align: center;}div.sk-container {display: inline-block;position: relative;}</style><div class="sk-top-container"><div class="sk-container"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="3d516887-0c68-4e12-a611-ff5b6a6e7288" type="checkbox" ><label class="sk-toggleable__label" for="3d516887-0c68-4e12-a611-ff5b6a6e7288">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[('slidingwindow', SlidingWindow(size=5, stride=2)),
                    ('featureunion',
                     FeatureUnion(transformer_list=[('window_mean',
                                                     FunctionTransformer(func=functools.partial(<function mean at 0x7efc8803a3a0>, axis=1, keepdims=True))),
                                                    ('window_variance',
                                                     FunctionTransformer(func=functools.partial(<function var at 0x7efc8803a700>, axis=1, keepdims=True))),
                                                    ('window_topology',
                                                     Pipeline(steps=[('takensembedding',
                                                                      TakensEmbedding()),
                                                                     ('vietorisripspersistence',
                                                                      VietorisRipsPersistence()),
                                                                     ('amplitude',
                                                                      Amplitude())]))])),
                    ('randomforestregressor', RandomForestRegressor())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="dcad77d3-ed5a-4a50-a772-2b61579ad777" type="checkbox" ><label class="sk-toggleable__label" for="dcad77d3-ed5a-4a50-a772-2b61579ad777">SlidingWindow</label><div class="sk-toggleable__content"><pre>SlidingWindow(size=5, stride=2)</pre></div></div></div><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="b0ee6a06-8e32-4613-ae1c-5d4bed118619" type="checkbox" ><label class="sk-toggleable__label" for="b0ee6a06-8e32-4613-ae1c-5d4bed118619">featureunion: FeatureUnion</label><div class="sk-toggleable__content"><pre>FeatureUnion(transformer_list=[('window_mean',
                                    FunctionTransformer(func=functools.partial(<function mean at 0x7efc8803a3a0>, axis=1, keepdims=True))),
                                   ('window_variance',
                                    FunctionTransformer(func=functools.partial(<function var at 0x7efc8803a700>, axis=1, keepdims=True))),
                                   ('window_topology',
                                    Pipeline(steps=[('takensembedding',
                                                     TakensEmbedding()),
                                                    ('vietorisripspersistence',
                                                     VietorisRipsPersistence()),
                                                    ('amplitude', Amplitude())]))])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>window_mean</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="757dac35-7983-4712-936d-91a52412bced" type="checkbox" ><label class="sk-toggleable__label" for="757dac35-7983-4712-936d-91a52412bced">FunctionTransformer</label><div class="sk-toggleable__content"><pre>FunctionTransformer(func=functools.partial(<function mean at 0x7efc8803a3a0>, axis=1, keepdims=True))</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>window_variance</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="37e8e9dc-01e0-46ad-a5ea-9e5bc8310961" type="checkbox" ><label class="sk-toggleable__label" for="37e8e9dc-01e0-46ad-a5ea-9e5bc8310961">FunctionTransformer</label><div class="sk-toggleable__content"><pre>FunctionTransformer(func=functools.partial(<function var at 0x7efc8803a700>, axis=1, keepdims=True))</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>window_topology</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="596bc45a-0ffc-42df-b0e0-8af767c8ddff" type="checkbox" ><label class="sk-toggleable__label" for="596bc45a-0ffc-42df-b0e0-8af767c8ddff">TakensEmbedding</label><div class="sk-toggleable__content"><pre>TakensEmbedding()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="4b0d8d19-61ba-4f32-b04f-7a0f305e90cc" type="checkbox" ><label class="sk-toggleable__label" for="4b0d8d19-61ba-4f32-b04f-7a0f305e90cc">VietorisRipsPersistence</label><div class="sk-toggleable__content"><pre>VietorisRipsPersistence()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="75d569da-5eae-4eb9-a63b-0dec7c674938" type="checkbox" ><label class="sk-toggleable__label" for="75d569da-5eae-4eb9-a63b-0dec7c674938">Amplitude</label><div class="sk-toggleable__content"><pre>Amplitude()</pre></div></div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="e02ac908-4309-4834-9cac-cd407755bff8" type="checkbox" ><label class="sk-toggleable__label" for="e02ac908-4309-4834-9cac-cd407755bff8">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor()</pre></div></div></div></div></div></div></div>



.. code:: ipython3

    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    score = pipe.score(X, y)
    y_pred, score




.. parsed-literal::

    (array([-4.34, -3.48, -1.58]), 0.8747)



Endogeneous target preparation with ``Labeller``
------------------------------------------------

Let us say that we simply wish to predict the future of a time series
from itself. This is very common in the study of financial markets for
example. ``giotto-tda`` provides convenience classes for target
preparation from a time series. This notebook only shows a very simple
example: many more options are described in ``Labeller``\ ’s
documentation.

If we wished to create a target ``y`` from ``X`` such that ``y[i]`` is
equal to ``X[i + 1]``, while also modifying ``X`` and ``y`` so that they
still have the same length, we could proceed as follows:

.. code:: ipython3

    from gtda.time_series import Labeller
    
    X = np.arange(10)
    
    Lab = Labeller(size=1, func=np.max)
    Xl, yl = Lab.fit_transform_resample(X, X)
    Xl, yl




.. parsed-literal::

    (array([0, 1, 2, 3, 4, 5, 6, 7, 8]), array([1, 2, 3, 4, 5, 6, 7, 8, 9]))



Notice that we are feeding two copies of ``X`` to
``fit_transform_resample`` in this case!

This is what fitting an end-to-end pipeline for future prediction using
topology could look like. Again, you are encouraged to include your own
non-topological features in the mix!

.. code:: ipython3

    SW = SlidingWindow(size=5)
    TE = TakensEmbedding(time_delay=1, dimension=2)
    VR = VietorisRipsPersistence()
    Ampl = Amplitude()
    RFR = RandomForestRegressor()
    
    # Full pipeline including the regressor
    pipe = make_pipeline(Lab, SW, TE, VR, Ampl, RFR)
    pipe




.. raw:: html

    <style>div.sk-top-container {color: black;background-color: white;}div.sk-toggleable {background-color: white;}label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}div.sk-estimator:hover {background-color: #d4ebff;}div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}div.sk-item {z-index: 1;}div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}div.sk-parallel-item:only-child::after {width: 0;}div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}div.sk-label-container {position: relative;z-index: 2;text-align: center;}div.sk-container {display: inline-block;position: relative;}</style><div class="sk-top-container"><div class="sk-container"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="c91731c7-2fa0-4ec3-8b36-ee6405e74e01" type="checkbox" ><label class="sk-toggleable__label" for="c91731c7-2fa0-4ec3-8b36-ee6405e74e01">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[('labeller',
                     Labeller(func=<function amax at 0x7efc880355e0>, size=1)),
                    ('slidingwindow', SlidingWindow(size=5)),
                    ('takensembedding', TakensEmbedding()),
                    ('vietorisripspersistence', VietorisRipsPersistence()),
                    ('amplitude', Amplitude()),
                    ('randomforestregressor', RandomForestRegressor())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="55b80ddc-ee51-4e93-9f1d-54f74a55ce73" type="checkbox" ><label class="sk-toggleable__label" for="55b80ddc-ee51-4e93-9f1d-54f74a55ce73">Labeller</label><div class="sk-toggleable__content"><pre>Labeller(func=<function amax at 0x7efc880355e0>, size=1)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="6ff3bb00-a6d5-459f-bec5-75fee46f7f60" type="checkbox" ><label class="sk-toggleable__label" for="6ff3bb00-a6d5-459f-bec5-75fee46f7f60">SlidingWindow</label><div class="sk-toggleable__content"><pre>SlidingWindow(size=5)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="303d337d-471d-455c-8717-98d2e8a89ec6" type="checkbox" ><label class="sk-toggleable__label" for="303d337d-471d-455c-8717-98d2e8a89ec6">TakensEmbedding</label><div class="sk-toggleable__content"><pre>TakensEmbedding()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="38785ab4-9b34-499e-8fd6-9041b1c83127" type="checkbox" ><label class="sk-toggleable__label" for="38785ab4-9b34-499e-8fd6-9041b1c83127">VietorisRipsPersistence</label><div class="sk-toggleable__content"><pre>VietorisRipsPersistence()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="b19a865e-e738-4171-b362-bc7ea41e559e" type="checkbox" ><label class="sk-toggleable__label" for="b19a865e-e738-4171-b362-bc7ea41e559e">Amplitude</label><div class="sk-toggleable__content"><pre>Amplitude()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="dd7975b8-0dec-46ef-b6f7-a730bc0ebd52" type="checkbox" ><label class="sk-toggleable__label" for="dd7975b8-0dec-46ef-b6f7-a730bc0ebd52">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor()</pre></div></div></div></div></div></div></div>



.. code:: ipython3

    pipe.fit(X, X)
    y_pred = pipe.predict(X)
    y_pred




.. parsed-literal::

    array([6.948, 6.948, 6.948, 6.948, 6.948])



Where to next?
--------------

1. There are two additional simple ``TransformerResamplerMixin``\ s in
   ``gtda.time_series``: ``Resampler`` and ``Stationarizer``.

2. The sort of pipeline for topological feature extraction using Takens
   embedding is a bit crude. More sophisticated methods exist for
   extracting robust topological summaries from (windows on) time
   series. A good source of inspiration is the following paper:

      `Persistent Homology of Complex Networks for Dynamic State
      Detection <https://arxiv.org/abs/1904.07403>`__, by A. Myers, E.
      Munch, and F. A. Khasawneh.

   The module ``gtda.graphs`` contains several transformers implementing
   the main algorithms proposed there.

3. Advanced users may be interested in ``ConsecutiveRescaling``, which
   can be found in ``gtda.point_clouds``.

4. The notebook `Case study: Lorenz
   attractor <https://github.com/giotto-ai/giotto-tda/blob/master/examples/lorenz_attractor.ipynb>`__
   is an advanced use-case for ``TakensEmbedding`` and other time series
   forecasting techniques inspired by topology.
