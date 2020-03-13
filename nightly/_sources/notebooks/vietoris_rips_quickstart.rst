Topological feature extraction using VietorisRips and PersistenceEntropy
========================================================================

In this notebook, we showcase the ease of use of one of the core
components of ``giotto-tda``: VietorisRipsPersistence, alongwith
vectorisation methods. We first list steps in a typical,
topological-feature extraction routine and then show to encapsulate them
with a standard ``scikit-learn``–like pipeline.

Import libraries
----------------

.. code:: ipython3

    from gtda.diagrams import PersistenceEntropy
    from gtda.homology import VietorisRipsPersistence
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    from datasets import generate_point_clouds

Generate data
-------------

Let’s begin by generating 3D point clouds of spheres and tori, along
with a label of 0 (1) for each sphere (torus). We also add noise to each
point cloud, whose effect is to displace the points sampling the
surfaces by a random amount in a random direction:

.. code:: ipython3

    point_clouds, labels = generate_point_clouds(100, 10, 0.1)

Calculate persistent homology
-----------------------------

Instantiate a VietorisRipsPersistence transformer and calculate
persistence diagrams for this collection of point clouds.

.. code:: ipython3

    vietorisrips_tr = VietorisRipsPersistence()
    diagrams = vietorisrips_tr.fit_transform(point_clouds)

Extract features
----------------

Instantiate a PersistenceEntropy transformer and extract features from
the persistence diagrams.

.. code:: ipython3

    entropy_tr = PersistenceEntropy()
    features = entropy_tr.fit_transform(diagrams)

Use the new features in a standard classifier
---------------------------------------------

Leverage the compatibility with scikit-learn to perform a train-test
split and score the features.

.. code:: ipython3

    X_train, X_valid, y_train, y_valid = train_test_split(features, labels)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    model.score(X_valid, y_valid)




.. parsed-literal::

    0.98



Encapsulates the steps above in a pipeline
------------------------------------------

Subdivide into train-validation first, and use the pipeline.

.. code:: ipython3

    from gtda.pipeline import make_pipeline

Define the pipeline
-------------------

Chain transformers from giotto-tda with scikit-learn ones.

.. code:: ipython3

    steps = [VietorisRipsPersistence(),
             PersistenceEntropy(),
             RandomForestClassifier()]
    pipeline = make_pipeline(*steps)

Prepare the data
----------------

Train-test split on the point-cloud data

.. code:: ipython3

    pcs_train, pcs_valid, labels_train, labels_valid = train_test_split(
        point_clouds, labels)

Train and score
---------------

.. code:: ipython3

    pipeline.fit(pcs_train, labels_train)
    pipeline.score(pcs_valid, labels_valid)




.. parsed-literal::

    1.0



