===============
Theory Glossary
===============

.. contents::
   :depth: 3
..

Symbols
=======

+------------------------------+--------------------------------------+
| :math:`\Bbbk`                | An arbitrary field.                  |
+------------------------------+--------------------------------------+
| :math:`\mathbb R`            | The field of real numbers.           |
+------------------------------+--------------------------------------+
| :math:`\overline{\mathbb R}` | The two point compactification       |
|                              | :math:`[-\infty, +\infty]` of the    |
|                              | real numbers.                        |
+------------------------------+--------------------------------------+
| :math:`\mathbb N`            | The counting numbers                 |
|                              | :math:`0,1,2, \ldots` as a subset of |
|                              | :math:`\mathbb R`.                   |
+------------------------------+--------------------------------------+
| :math:`\mathbb R^d`          | The vector space of :math:`d`-tuples |
|                              | of real numbers.                     |
+------------------------------+--------------------------------------+
| :math:`\Delta`               | The multiset                         |
|                              | :math:`\lbrace (                     |
|                              | s, s) \mid s \in \mathbb{R} \rbrace` |
|                              | with multiplicity                    |
|                              | :math:`( s,s ) \mapsto +\infty`.     |
+------------------------------+--------------------------------------+

Homology
========

.. _cubical_complex:

Cubical complex
---------------

An *elementary interval* :math:`I_a` is a subset of :math:`\mathbb{R}`
of the form :math:`[a, a+1]` or :math:`[a,a] = \{a\}` for some
:math:`a \in \mathbb{R}`. These two types are called respectively
*non-degenerate* and *degenerate*. To a non-degenerate elementary
interval we assign two degenerate elementary intervals

.. math:: d^+ I_a = \lbrack a+1, a+1 \rbrack \qquad \text{and} \qquad d^- I_a = \lbrack a, a \rbrack.

An *elementary cube* is a subset of the form

.. math:: I_{a_1} \times \cdots \times I_{a_N} \subset \mathbb{R}^N

where each :math:`I_{a_i}` is an elementary interval. We refer to the
total number of its non-degenerate factors
:math:`I_{a_{k_1}}, \dots, I_{a_{k_n}}` as its *dimension* and, assuming

.. math:: a_{k_1} < \cdots < a_{k_{n,}}

we define for :math:`i = 1, \dots, n` the following two elementary cubes

.. math:: d_i^\pm I^N = I_{a_1} \times \cdots \times d^\pm I_{a_{k_i}} \times \cdots \times I_{a_{N.}}

A *cubical complex* is a finite set of elementary cubes of
:math:`\mathbb{R}^N`, and a *subcomplex* of :math:`X` is a cubical
complex whose elementary cubes are also in :math:`X`. We denote the set
of :math:`n`-dimensional cubes as :math:`X_n`.

Reference:
~~~~~~~~~~

(Kaczynski, Mischaikow, and Mrozek 2004)

.. _simplicial_complex:

Simplicial complex
------------------

A set :math:`\{v_0, \dots, v_n\} \subset \mathbb{R}^N` is said to be
*geometrically independent* if the vectors
:math:`\{v_0-v_1, \dots, v_0-v_n\}` are linearly independent. In this
case, we refer to their convex closure as a *simplex*, explicitly

.. math:: \lbrack v_0, \dots , v_n \rbrack = \left\{ \sum c_i (v_0 - v_i)\ \big|\ c_1+\dots+c_n = 1,\ c_i \geq 0 \right\}

and to :math:`n` as its *dimension*. The :math:`i`\ *-th face* of
:math:`\lbrack v_0, \dots, v_n \rbrack` is defined by

.. math:: d_i[v_0, \ldots, v_n] = [v_0, \dots, \widehat{v}_i, \dots, v_n]

where :math:`\widehat{v}_i` denotes the absence of :math:`v_i` from the
set.

A *simplicial complex* :math:`X` is a finite union of simplices in
:math:`\mathbb{R}^N` satisfying that every face of a simplex in
:math:`X` is in :math:`X` and that the non-empty intersection of two
simplices in :math:`X` is a face of each. Every simplicial complex
defines an abstract simplicial complex.

.. _abstract_simplicial_complex:

Abstract simplicial complex
---------------------------

An *abstract simplicial complex* is a pair of sets :math:`(V, X)` with
the elements of :math:`X` being subsets of :math:`V` such that:

#. for every :math:`v` in :math:`V`, the singleton :math:`\{v\}` is in
   :math:`X` and

#. if :math:`x` is in :math:`X` and :math:`y` is a subset of :math:`x`,
   then :math:`y` is in :math:`X`.

We abuse notation and denote the pair :math:`(V, X)` simply by
:math:`X`.

The elements of :math:`X` are called *simplices* and the *dimension* of
a simplex :math:`x` is defined by :math:`|x| = \# x - 1` where
:math:`\# x` denotes the cardinality of :math:`x`. Simplices of
dimension :math:`d` are called :math:`d`-simplices. We abuse terminology
and refer to the elements of :math:`V` and to their associated
:math:`0`-simplices both as *vertices*.

The :math:`k`\ *-skeleton* :math:`X_k` of a simplicial complex :math:`X`
is the subcomplex containing all simplices of dimension at most
:math:`k`. A simplicial complex is said to be :math:`d`\ *-dimensional*
if :math:`d` is the smallest integer satisfying :math:`X = X_d`.

A *simplicial map* between simplicial complexes is a function between
their vertices such that the image of any simplex via the induced map is
a simplex.

A simplicial complex :math:`X` is a *subcomplex* of a simplicial complex
:math:`Y` if every simplex of :math:`X` is a simplex of :math:`Y`.

Given a finite abstract simplicial complex :math:`X = (V, X)` we can
choose a bijection from :math:`V` to a geometrically independent subset
of :math:`\mathbb R^N` and associate a simplicial complex to :math:`X`
called its *geometric realization*.

.. _ordered_simplical_complex:

Ordered simplicial complex
--------------------------

An *ordered simplicial complex* is an abstract simplicial complex where
the set of vertices is equipped with a partial order such that the
restriction of this partial order to any simplex is a total order. We
denote an :math:`n`-simplex using its ordered vertices by
:math:`\lbrack v_0, \dots, v_n \rbrack`.

A *simplicial map* between ordered simplicial complexes is a simplicial
map :math:`f` between their underlying simplicial complexes preserving
the order, i.e., :math:`v \leq w` implies :math:`f(v) \leq f(w)`.

.. _directed_simplicial_complex:

Directed simplicial complex
---------------------------

A *directed simplicial complex* is a pair of sets :math:`(V, X)` with
the elements of :math:`X` being tuples of elements of :math:`V`, i.e.,
elements in :math:`\bigcup_{n\geq1} V^{\times n}` such that:

#. for every :math:`v` in :math:`V`, the tuple :math:`v` is in :math:`X`

#. if :math:`x` is in :math:`X` and :math:`y` is a subtuple of
   :math:`x`, then :math:`y` is in :math:`X`.

With appropriate modifications the same terminology and notation
introduced for ordered simplicial complex applies to directed simplicial
complex.

.. _chain_complex:

Chain complex
-------------

A *chain complex* of is a pair :math:`(C_*, \partial)` where

.. math:: C_* = \bigoplus_{n \in \mathbb Z} C_n \quad \mathrm{and} \quad \partial = \bigoplus_{n \in \mathbb Z} \partial_n

with :math:`C_n` a :math:`\Bbbk`-vector space and
:math:`\partial_n : C_{n+1} \to C_n` is a :math:`\Bbbk`-linear map such
that :math:`\partial_{n+1} \partial_n = 0`. We refer to :math:`\partial`
as the *boundary map* of the chain complex.

The elements of :math:`C` are called *chains* and if :math:`c \in C_n`
we say its *degree* is :math:`n` or simply that it is an
:math:`n`-chain. Elements in the kernel of :math:`\partial` are called
*cycles*, and elements in the image of :math:`\partial` are called
*boundaries*. Notice that every boundary is a cycle. This fact is
central to the definition of homology.

A *chain map* is a :math:`\Bbbk`-linear map :math:`f : C \to C'` between
chain complexes such that :math:`f(C_n) \subseteq C'_n` and
:math:`\partial f = f \partial`.

Given a chain complex :math:`(C_*, \partial)`, its linear dual
:math:`C^*` is also a chain complex with
:math:`C^{-n} = \mathrm{Hom_\Bbbk}(C_n, \Bbbk)` and boundary map
:math:`\delta` defined by :math:`\delta(\alpha)(c) = \alpha(\partial c)`
for any :math:`\alpha \in C^*` and :math:`c \in C_*`.

.. _homology_and_cohomology:

Homology and cohomology
-----------------------

Let :math:`(C_*, \partial)` be a chain complex. Its :math:`n`\ *-th
homology group* is the quotient of the subspace of :math:`n`-cycles by
the subspace of :math:`n`-boundaries, that is,
:math:`H_n(C_*) = \mathrm{ker}(\partial_n)/ \mathrm{im}(\partial_{n+1})`.
The *homology* of :math:`(C, \partial)` is defined by
:math:`H_*(C) = \bigoplus_{n \in \mathbb Z} H_n(C)`.

When the chain complex under consideration is the linear dual of a chain
complex we sometimes refer to its homology as the *cohomology* of the
predual complex and write :math:`H^n` for :math:`H_{-n}`.

A chain map :math:`f : C \to C'` induces a map between the associated
homologies.

.. _simplicial_chains_and_simplicial_homology:

Simplicial chains and simplicial homology
-----------------------------------------

Let :math:`X` be an ordered or directed simplicial complex. Define its
*simplicial chain complex with* :math:`\Bbbk`\ *-coefficients*
:math:`C_*(X; \Bbbk)` by

.. math:: C_n(X; \Bbbk) = \Bbbk\{X_n\}, \qquad \partial_n(x) = \sum_{i=0}^{n} (-1)^i d_ix

and its *homology and cohomology with* :math:`\Bbbk`\ *-coefficients* as
the homology and cohomology of this chain complex. We use the notation
:math:`H_*(X; \Bbbk)` and :math:`H^*(X; \Bbbk)` for these.

A simplicial map induces a chain map between the associated simplicial
chain complexes and, therefore, between the associated simplicial
(co)homologies.

.. _cubical_chains_and_cubical_homology:

Cubical chains and cubical homology
-----------------------------------

Let :math:`X` be a cubical complex. Define its *cubical chain complex
with* :math:`\Bbbk`\ *-coefficients* :math:`C_*(X; \Bbbk)` by

.. math:: C_n(X; \Bbbk) = \Bbbk\{X_n\}, \qquad \partial_n x = \sum_{i = 1}^{n} (-1)^{i-1}(d^+_i x - d^-_i x)

where :math:`x = I_1 \times \cdots \times I_N` and :math:`s(i)` is the
dimension of :math:`I_1 \times \cdots \times I_i`. Its *homology and
cohomology with* :math:`\Bbbk`\ *-coefficients* is the homology and
cohomology of this chain complex. We use the notation
:math:`H_*(X; \Bbbk)` and :math:`H^*(X; \Bbbk)` for these.

.. _filtered_complex:

Filtered complex
----------------

A *filtered complex* is a collection of simplicial or cubical complexes
:math:`\{X_s\}_{s \in \mathbb R}` such that :math:`X_s` is a subcomplex
of :math:`X_t` for each :math:`s \leq t`.

.. _cellwise_filtration:

Cellwise filtration
-------------------

A *cellwise filtration* is a simplicial or cubical complex :math:`X`
together with a total order :math:`\leq` on its simplices or elementary
cubes such that for each :math:`y \in X` the set
:math:`\{x \in X\ :\ x \leq y\}` is a subcomplex of :math:`X`. A
cellwise filtration can be naturally thought of as a filtered complex.

.. _clique_and_flag_complexes:

Clique and flag complexes
-------------------------

Let :math:`G` be a :math:`1`-dimensional abstract (resp. directed)
simplicial complex. The abstract (resp. directed) simplicial complex
:math:`\langle G \rangle` has the same set of vertices as :math:`G` and
:math:`\{v_0, \dots, v_n\}` (resp. :math:`(v_0, \dots, v_n)`) is a
simplex in :math:`\langle G \rangle` if an only if :math:`\{v_i, v_j\}`
(resp. :math:`(v_i, v_j)`) is in :math:`G` for each pair of vertices
:math:`v_i, v_j`.

An abstract (resp. directed) simplicial complex :math:`X` is a *clique
(resp. flag) complex* if :math:`X = \langle G \rangle` for some
:math:`G`.

Given a function

.. math:: w : G \to \mathbb R \cup \{\infty\}

consider the extension

.. math:: w : \langle G \rangle \to \mathbb R \cup \{\infty\}

defined respectively by

.. math::

   \begin{aligned}
       w\{v_0, \dots, v_n\} & = \max\{ w\{v_i, v_j\}\ |\ i \neq j\} \\
       w(v_0, \dots, v_n) & = \max\{ w(v_i, v_j)\ |\ i < j\}
       \end{aligned}

and define the filtered complex
:math:`\{\langle G \rangle_{s}\}_{s \in \mathbb R}` by

.. math:: \langle G \rangle_s = \{\sigma \in \langle G \rangle\ |\ w(\sigma) \leq s\}.

A filtered complex :math:`\{X_s\}_{s \in \mathbb R}` is a *filtered
clique (resp. flag) complex* if :math:`X_s = \langle G \rangle_s` for
some :math:`(G,w)`.

.. _persistence_module:

Persistence module
------------------

A *persistence module* is a collection containing a :math:`\Bbbk`-vector
spaces :math:`V(s)` for each real number :math:`s` together with
:math:`\Bbbk`-linear maps :math:`f_{st} : V(s) \to V(t)`, referred to as
*structure maps*, for each pair :math:`s \leq t`, satisfying naturality,
i.e., if :math:`r \leq s \leq t`, then
:math:`f_{rt} = f_{st} \circ f_{rs}` and tameness, i.e., all but
finitely many structure maps are isomorphisms.

A *morphism of persistence modules* :math:`F : V \to W` is a collection
of linear maps :math:`F(s) : V(s) \to W(s)` such that
:math:`F(t) \circ f_{st} = f_{st} \circ F(s)` for each par of reals
:math:`s \leq t`. We say that :math:`F` is an *isomorphisms* if each
:math:`F(s)` is.

.. _persistent_simplicial_(co)homology:

Persistent simplicial (co)homology
----------------------------------

Let :math:`\{X(s)\}_{s \in \mathbb R}` be a set of ordered or directed
simplicial complexes together with simplicial maps
:math:`f_{st} : X(s) \to X(t)` for each pair :math:`s \leq t`, such that

.. math:: r \leq s \leq t\ \quad\text{implies} \quad f_{rt} = f_{st} \circ f_{rs}

for example, a filtered complex. Its *persistent simplicial homology
with* :math:`\Bbbk`\ *-coefficients* is the persistence module

.. math:: H_*(X(s); \Bbbk)

with structure maps
:math:`H_*(f_{st}) : H_*(X(s); \Bbbk) \to H_*(X(t); \Bbbk)` induced form
the maps :math:`f_{st.}` In general, the collection constructed this way
needs not satisfy the tameness condition of a persistence module, but we
restrict attention to the cases where it does. Its *persistence
simplicial cohomology with* :math:`\Bbbk`\ *-coefficients* is defined
analogously.

.. _vietoris-rips_complex_and_vietoris-rips_persistence:

Vietoris-Rips complex and Vietoris-Rips persistence
---------------------------------------------------

Let :math:`(X, d)` be a finite metric space. Define the Vietoris-Rips
complex of :math:`X` as the filtered complex :math:`VR_s(X)` that
contains a subset of :math:`X` as a simplex if all pairwise distances in
the subset are less than or equal to :math:`s`, explicitly

.. math:: VR_s(X) = \Big\{ \lbrack v_0,\dots,v_n \rbrack \ \Big|\ \forall i,j\ \,d(v_i, v_j) \leq s \Big\}.

The *Vietoris-Rips persistence* of :math:`(X, d)` is the persistent
simplicial (co)homology of :math:`VR_s(X)`.

A more general version is obtained by replacing the distance function
with an arbitrary function

.. math:: w : X \times X \to \mathbb R \cup \{\infty\}

and defining :math:`VR_s(X)` as the filtered clique complex associated
to :math:`(X \times X ,w)`.

.. _cech_complex_and_cech_persistence:

Čech complex and Čech persistence
---------------------------------

Let :math:`(X, d)` be a point cloud. Define the Čech complex of
:math:`X` as the filtered complex :math:`\check{C}_s(X)` that is empty
if :math:`s<0` and, if :math:`s \geq 0`, contains a subset of :math:`X`
as a simplex if the balls of radius :math:`s` with centers in the subset
have a non-empty intersection, explicitly

.. math:: \check{C}_s(X) = \Big\{ \lbrack v_0,\dots,v_n \rbrack \ \Big|\ \bigcap_{i=0}^n B_s(x_i) \neq \emptyset \Big\}.

The *Čech persistence (co)homology* of :math:`(X,d)` is the persistent
simplicial (co)homology of :math:`\check{C}_s(X)`.

Multiset
--------

A *multiset* is a pair :math:`(S, \phi)` where :math:`S` is a set and
:math:`\phi : S \to \mathbb N \cup \{+\infty\}` is a function attaining
positive values. For :math:`s \in S` we refer to :math:`\phi(s)` as its
*multiplicity*. The *union* of two multisets
:math:`(S_1, \phi_1), (S_2, \phi_2)` is the multiset
:math:`(S_1 \cup S_2, \phi_1 \cup \phi_2)` with

.. math::

   (\phi_1 \cup \phi_2)(s) = 
       \begin{cases}
       \phi_1(s) & s \in S_1, s \not\in S_2 \\
       \phi_2(s) & s \in S_2, s \not\in S_1 \\
       \phi_1(s) + \phi_2(s) & s \in S_1, s \in S_2. \\
       \end{cases}

.. _persistence_diagram:

Persistence diagram
-------------------

A *persistence diagram* is a multiset of points in

.. math:: \mathbb R \times \big( \mathbb{R} \cup \{+\infty\} \big).

Given a persistence module, its associated persistence diagram is
determined by the following condition: for each pair :math:`s,t` the
number counted with multiplicity of points :math:`(b,d)` in the
multiset, satisfying :math:`b \leq s \leq t < d` is equal to the rank of
:math:`f_{st.}`

A well known result establishes that there exists an isomorphism between
two persistence module if and only if their persistence diagrams are
equal.

.. _wasserstein_and_bottleneck_distance:

Wasserstein and bottleneck distance
-----------------------------------

The :math:`p`\ *-Wasserstein distance* between two persistence diagrams
:math:`D_1` and :math:`D_2` is the infimum over all bijections
:math:`\gamma: D_1 \cup \Delta \to D_2 \cup \Delta` of

.. math:: \Big(\sum_{x \in D_1 \cup \Delta} ||x - \gamma(x)||_\infty^p \Big)^{1/p}

where :math:`||-||_\infty` is defined for :math:`(x,y) \in \mathbb R^2`
by :math:`\max\{|x|, |y|\}`.

The limit :math:`p \to \infty` defines the *bottleneck distance*. More
explicitly, it is the infimum over the same set of bijections of the
value

.. math:: \sup_{x \in D_1 \cup \Delta} ||x - \gamma(x)||_{\infty.}

The set of persistence diagrams together with any of the distances above
is a metric space.

.. _reference-1:

Reference:
~~~~~~~~~~

(Kerber, Morozov, and Nigmetov 2017)

.. _persistence_landscape:

Persistence landscape
---------------------

Let :math:`\{(b_i, d_i)\}_{i \in I}` be a persistence diagram. Its
*persistence landscape* is the set
:math:`\{\lambda_k\}_{k \in \mathbb N}` of functions

.. math:: \lambda_k : \mathbb R \to \overline{\mathbb R}

defined by letting :math:`\lambda_k(t)` be the :math:`k`-th largest
value of the set :math:`\{\Lambda_i(t)\}_ {i \in I}` where

.. math:: \Lambda_i(t) = \left[ \min \{t-b_i, d_i-t\}\right]_+

and :math:`c_+ := \max(c,0)`. The function :math:`\lambda_k` is referred
to as the *:math:`k`-layer of the persistence landscape*.

We describe the graph of each :math:`\lambda_k` intuitively. For each
:math:`i \in I`, draw an isosceles triangle with base the interval
:math:`(b_i, d_i)` on the horizontal :math:`t`-axis, and sides with
slope 1 and :math:`-1`. This subdivides the plane into a number of
polygonal regions. Label each of these regions by the number of
triangles containing it. If :math:`P_k` is the union of the polygonal
regions with values at least :math:`k`, then the graph of
:math:`\lambda_k` is the upper contour of :math:`P_k`, with
:math:`\lambda_k(a) = 0` if the vertical line :math:`t=a` does not
intersect :math:`P_k`.

The persistence landscape construction defines a vectorization of the
set of persistence diagrams with target the vector space of real-valued
function on :math:`\mathbb N \times \mathbb R`. For any
:math:`p = 1,\dots,\infty` we can restrict attention to persistence
diagrams :math:`D` whose associated persistence landscape
:math:`\lambda` is :math:`p`-integrable, that is to say,

.. math::

   \label{equation:persistence_landscape_norm}    
       ||\lambda||_p = \left( \sum_{i \in \mathbb N} ||\lambda_i||^p_p \right)^{1/p}

where

.. math:: ||\lambda_i||_p = \left( \int_{\mathbb R} \lambda_i^p(x)\, dx \right)^{1/p}

is finite. In this case we refer to
`[equation:persistence_landscape_norm] <#equation:persistence_landscape_norm>`__
as the *landscape* :math:`p`-*amplitude* of :math:`D`.

References:
~~~~~~~~~~~

(Bubenik 2015)

.. _weighted_silhouette:

Weighted silhouette
-------------------

Let :math:`D = \{(b_i, d_i)\}_{i \in I}` be a persistence diagram and
:math:`w = \{w_i\}_{i \in I}` a set of positive real numbers. The
*silhouette of :math:`D` weighted by :math:`w`* is the function
:math:`\phi : \mathbb R \to \mathbb R` defined by

.. math:: \phi(t) = \frac{\sum_{i \in I}w_i \Lambda_i(t)}{\sum_{i \in I}w_i},

where

.. math:: \Lambda_i(t) = \left[ \min \{t-b_i, d_i-t\}\right]_+

and :math:`c_+ := \max(c,0)`. When :math:`w_i = \vert d_i - b_i \vert^p`
for :math:`0 < p \leq \infty` we refer to :math:`\phi` as the
*:math:`p`-power-weighted silhouette* of :math:`D`. The silhouette
construction defines a vectorization of the set of persistence diagrams
with target the vector space of continuous real-valued functions on
:math:`\mathbb R`.

.. _references-1:

References:
~~~~~~~~~~~

(Chazal et al. 2014)

.. _persistence_entropy:

Persistence entropy
-------------------

Intuitively, this is a measure of the entropy of the points in a
persistence diagram. Precisely, let :math:`D = \{(b_i, d_i)\}_{i \in I}`
be a persistence diagram with each :math:`d_i < +\infty`. The
*persistence entropy* of :math:`D` is defined by

.. math:: E(D) = - \sum_{i \in I} p_i \log(p_i)

where

.. math:: p_i = \frac{(d_i - b_i)}{L_D} \qquad \text{and} \qquad L_D = \sum_{i \in I} (d_i - b_i) .

.. _references-2:

References:
~~~~~~~~~~~

(Rucco et al. 2016)

.. _betti_curve:

Betti curve
-----------

Let :math:`D` be a persistence diagram. Its *Betti curve* is the
function :math:`\beta_D : \mathbb R \to \mathbb N` whose value on
:math:`s \in \mathbb R` is the number, counted with multiplicity, of
points :math:`(b_i,d_i)` in :math:`D` such that :math:`b_i \leq s <d_i`.

The name is inspired from the case when the persistence diagram comes
from persistent homology.

.. _metric_space:

Metric space
------------

A set :math:`X` with a function

.. math:: d : X \times X \to \mathbb R

is said to be a *metric space* if the values of :math:`d` are all
non-negative and for all :math:`x,y,z \in X`

.. math:: d(x,y) = 0\ \Leftrightarrow\ x = y

.. math:: d(x,y) = d(y,x)

.. math:: d(x,z) \leq d(x,y) + d(y, z).

In this case the :math:`d` is referred to as the *metric* or the
*distance function*.

.. _inner_product_and_norm:

Inner product and norm
----------------------

A vector space :math:`V` together with a function

.. math:: \langle -, - \rangle : V \times V \to \mathbb R

is said to be an *inner product space* if for all :math:`u,v,w \in V`
and :math:`a \in \mathbb R`

.. math:: u \neq 0\ \Rightarrow\ \langle u, u \rangle > 0

.. math:: \langle u, v\rangle = \langle v, u\rangle

.. math:: \langle au+v, w \rangle = a\langle u, w \rangle + \langle v, w \rangle.

The function :math:`\langle -, - \rangle` is referred to as the *inner
product*.

A vector space :math:`V` together with a function

.. math:: ||-|| : V \to \mathbb R

is said to be an *normed space* if the values of :math:`||-||` are all
non-negative and for all :math:`u,v \in V` and :math:`a \in \mathbb R`

.. math:: ||v|| = 0\ \Leftrightarrow\ u = 0

.. math:: ||a u || = |a|\, ||u||

.. math:: ||u+v|| = ||u|| + ||v||.

The function :math:`||-||` is referred to as the *norm*.

An inner product space is naturally a norm space with

.. math:: ||u|| = \sqrt{\langle u, u \rangle}

and a norm space is naturally a metric space with distance function

.. math:: d(u,v) = ||u-v||.

.. _euclidean_distance_and_norm:

Euclidean distance and norm
---------------------------

The vector space :math:`\mathbb R^n` is an inner product space with
inner product

.. math:: \langle x, y \rangle = (x_1-y_1)^2 + \cdots + (x_n-y_n)^2.

This inner product is referred to as *dot product* and the associated
norm and distance function are respectively named *euclidean norm* and
*euclidean distance*.

.. _vectorization_kernel_and_amplitude:

Vectorization, kernel and amplitude
-----------------------------------

Let :math:`X` be a set, for example, the set of all persistence
diagrams. A *vectorization* for :math:`X` is a function

.. math:: \phi : X \to V

where :math:`V` is a vector space. A *kernel* on the set :math:`X` is a
function

.. math:: k : X \times X \to \mathbb R

for which there exists a vectorization :math:`\phi : X \to V` with
:math:`V` an inner product space such that

.. math:: k(x,y) = \langle \phi(x), \phi(y) \rangle

for each :math:`x,y \in X`. Similarly, an *amplitude* on :math:`X` is a
function

.. math:: A : X \to \mathbb R

for which there exists a vectorization :math:`\phi : X \to V` with
:math:`V` a normed space such that

.. math:: A(x) = ||\phi(x)||

for all :math:`x \in X`.

.. _finite_metric_spaces_and_point_clouds:

Finite metric spaces and point clouds
-------------------------------------

A *finite metric space* is a finite set together with a metric. A
*distance matrix* associated to a finite metric space is obtained by
choosing a total order on the finite set and setting the
:math:`(i,j)`-entry to be equal to the distance between the :math:`i`-th
and :math:`j`-th elements.

A *point cloud* is a finite subset of :math:`\mathbb{R}^n` (for some
:math:`n`) together with the metric induced from the euclidean distance.

Time series
===========

.. _time_series:

Time series
-----------

A *time series* is a sequence :math:`\{y_i\}_{i = 0}^n` of real numbers.

A common construction of a times series :math:`\{x_i\}_{i = 0}^n` is
given by choosing :math:`x_0` arbitrarily as well as a step parameter
:math:`h` and setting

.. math:: x_i = x_0 + h\cdot i.

Another usual construction is as follows: given a time series
:math:`\{x_i\}_{i = 0}^n \subseteq U` and a function

.. math:: f : U \subseteq \mathbb R \to \mathbb R

we obtain a new time series :math:`\{f(x_i)\}_{i = 0.}^n`

Generalizing the previous construction we can define a time series from
a function

.. math:: \varphi : U \times M \to M, \qquad U \subseteq \mathbb R, \qquad M \subseteq \mathbb R^d

using a function :math:`f : M \to \mathbb R` as follows: let
:math:`\{t_i\}_{i=0}^n` be a time series taking values in :math:`U`,
then

.. math:: \{f(\varphi(t_i, m))\}_{i=0}^n

for an arbitrarily chosen :math:`m \in M`.

.. _takens_embedding:

Takens embedding
----------------

Let :math:`M \subset \mathbb R^d` be a compact manifold of dimension
:math:`n`. Let

.. math:: \varphi : \mathbb R \times M \to M

and

.. math:: f : M \to \mathbb R

be generic smooth functions. Then, for any :math:`\tau > 0` the map

.. math:: M \to \mathbb R^{2n+1}

defined by

.. math:: x \mapsto\big( f(x), f(x_1), f(x_2), \dots, f(x_{2n}) \big)

where

.. math:: x_i = \varphi(i \cdot \tau, x)

is an injective map with full rank.

.. _reference-2:

Reference:
~~~~~~~~~~

(Takens 1981)

Manifold
--------

Intuitively, a manifold of dimension :math:`n` is a space locally
equivalent to :math:`\mathbb R^n`. Formally, a subset :math:`M` of
:math:`\mathbb R^d` is an :math:`n`-dimensional manifold if for each
:math:`x \in M` there exists an open ball
:math:`B(x) = \{ y \in M\,;\ d(x,y) < \epsilon\}` and a smooth function
with smooth inverse

.. math:: \phi_x : B(x) \to \{v \in \mathbb R^n\,;\ ||v||<1\}.

.. _references-3:

References:
~~~~~~~~~~~

(Milnor and Weaver 1997; Guillemin and Pollack 2010)

.. _compact_subset:

Compact subset
--------------

A subset :math:`K` of a metric space :math:`(X,d)` is said to be
*bounded* if there exist a real number :math:`D` such that for each pair
of elements in :math:`K` the distance between them is less than
:math:`D`. It is said to be *complete* if for any :math:`x \in X` it is
the case that :math:`x \in K` if for any :math:`\epsilon > 0` the
intersection between :math:`K` and :math:`\{y \,;\ d(x,y) < \epsilon \}`
is not empty. It is said to be *compact* if it is both bounded and
complete.

Bibliography
============

.. container:: references hanging-indent
   :name: refs

   .. container::
      :name: ref-bubenik2015statistical

      Bubenik, Peter. 2015. “Statistical Topological Data Analysis Using
      Persistence Landscapes.” *The Journal of Machine Learning
      Research* 16 (1): 77–102.

   .. container::
      :name: ref-chazal2014stochastic

      Chazal, Frédéric, Brittany Terese Fasy, Fabrizio Lecci, Alessandro
      Rinaldo, and Larry Wasserman. 2014. “Stochastic Convergence of
      Persistence Landscapes and Silhouettes.” In *Proceedings of the
      Thirtieth Annual Symposium on Computational Geometry*, 474–83.
      SOCG’14. Kyoto, Japan: Association for Computing Machinery.
      https://doi.org/10.1145/2582112.2582128.

   .. container::
      :name: ref-guillemin2010differential

      Guillemin, Victor, and Alan Pollack. 2010. *Differential
      Topology*. Vol. 370. American Mathematical Soc.

   .. container::
      :name: ref-mischaikow04computational

      Kaczynski, Tomasz, Konstantin Mischaikow, and Marian Mrozek. 2004.
      *Computational Homology*. Vol. 157. Applied Mathematical Sciences.
      Springer-Verlag, New York. https://doi.org/10.1007/b97315.

   .. container::
      :name: ref-kerber2017geometry

      Kerber, Michael, Dmitriy Morozov, and Arnur Nigmetov. 2017.
      “Geometry Helps to Compare Persistence Diagrams.” *Journal of
      Experimental Algorithmics (JEA)* 22: 1–4.

   .. container::
      :name: ref-milnor1997topology

      Milnor, John Willard, and David W Weaver. 1997. *Topology from the
      Differentiable Viewpoint*. Princeton university press.

   .. container::
      :name: ref-rucco2016characterisation

      Rucco, Matteo, Filippo Castiglione, Emanuela Merelli, and Marco
      Pettini. 2016. “Characterisation of the Idiotypic Immune Network
      Through Persistent Entropy.” In *Proceedings of Eccs 2014*,
      117–28. Springer.

   .. container::
      :name: ref-takens1981detecting

      Takens, Floris. 1981. “Detecting Strange Attractors in
      Turbulence.” In *Dynamical Systems and Turbulence, Warwick 1980*,
      366–81. Springer.
