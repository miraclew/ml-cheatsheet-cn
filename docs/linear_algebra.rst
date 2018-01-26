.. _linear_algebra:

==============
线性代数
==============

.. contents:: :local:

线性代数是一个数学工具帮助我们同时处理一组数字。
它提供了向量和矩阵(电子表格)这样的结构来存放这些数字，还有一些加减乘除的新的规则。
这里是线性代数一个简单概览 `post <https://medium.com/p/cd67aba4526c>`_.

向量
=======

向量是数字或术语的一维数组。在几何中，向量存储一个潜在的变化的大小和方向。向量[3, -2] 表示向右为3向下为2。多于一维的向量成为矩阵。


符号
--------
有一些不同的向量表示方法。这里是一些阅读时我们会碰到的。

.. math::

  v = \begin{bmatrix}
  1 \\
  2 \\
  3 \\
  \end{bmatrix}
  =
  \begin{pmatrix}
  1 \\
  2 \\
  3 \\
  \end{pmatrix}
  =
  \begin{bmatrix}
  1 & 2 & 3\\
  \end{bmatrix}


几何中的向量
-------------------

向量通常表示从一个点的运动。它们存储了潜在移动的大小和方向。向量[-2,5]表示向左2个单位，向上5个单位的移动 [1]_。

.. image:: images/vectors_geometry.png
    :align: center

向量可以被运用到空间中的任意点。这个向量的方向等于斜边的斜率。它的大小等于斜边的长度。


标量运算
-----------------
标量运算包含一个向量和一个数字。你可以通过加减乘以这个数字来修改一个向量中的所有值。

.. math::

  \begin{bmatrix}
  2 \\
  2 \\
  2 \\
  \end{bmatrix}
  +
  1
  =
  \begin{bmatrix}
  3 \\
  3 \\
  3 \\
  \end{bmatrix}


元素运算
----------------------

在元素运算如加，减，除，值的位置保持不变成为新的向量。向量A的第一个值对应向量B的第一个值，第二个值对应第二个值，以此类推。
这意味着完成运算后向量应该有相同的维度。*


.. math::

  \begin{bmatrix}
  a_1 \\
  a_2 \\
  \end{bmatrix}
  +
  \begin{bmatrix}
  b_1 \\
  b_2 \\
  \end{bmatrix}
  =
  \begin{bmatrix}
  a_1+b_1 \\
  a_2+b_2 \\
  \end{bmatrix}

::

  y = np.array([1,2,3])
  x = np.array([2,3,4])
  y + x = [3, 5, 7]
  y - x = [-1, -1, -1]
  y / x = [.5, .67, .75]

在下面查看numpy里broadcasting的详情。


点积
-----------

2个向量点积是一个矢量。向量和矩阵的点积是深度学习中最重要的运算。

.. math::

  \begin{bmatrix}
  a_1 \\
  a_2 \\
  \end{bmatrix}
  \cdot
  \begin{bmatrix}
  b_1 \\
  b_2 \\
  \end{bmatrix}
  = a_1 b_1+a_2 b_2

::

  y = np.array([1,2,3])
  x = np.array([2,3,4])
  np.dot(y,x) = 20


Hadamard 乘积
----------------
Hadamard 乘积是一个元素乘积，输出一个向量.

.. math::

  \begin{bmatrix}
  a_1 \\
  a_2 \\
  \end{bmatrix}
   \odot
  \begin{bmatrix}
  b_1 \\
  b_2 \\
  \end{bmatrix}
  =
  \begin{bmatrix}
  a_1 \cdot b_1 \\
  a_2 \cdot b_2 \\
  \end{bmatrix}

::

  y = np.array([1,2,3])
  x = np.array([2,3,4])
  y * x = [2, 6, 12]

向量场
-------------
A vector field shows how far the point (x,y) would hypothetically move if we applied a vector function 
to it like addition or multiplication. Given a point in space, a vector field shows the power and 
direction of our proposed change at a variety of points in a graph [2]_.
向量场显示

.. image:: images/vector_field.png
    :align: center

This vector field is an interesting one since it moves in different directions depending the starting point. The reason is that the vector behind this field stores terms like :math:`2x` or :math:`x^2` instead of scalar values like -2 and 5. For each point on the graph, we plug the x-coordinate into :math:`2x` or :math:`x^2` and draw an arrow from the starting point to the new location. Vector fields are extremely useful for visualizing machine learning techniques like Gradient Descent.


Matrices
========

A matrix is a rectangular grid of numbers or terms (like an Excel spreadsheet) with special rules for addition, subtraction, and multiplication.

Dimensions
----------
We describe the dimensions of a matrix in terms of rows by columns.

.. math::

  \begin{bmatrix}
  2 & 4 \\
  5 & -7 \\
  12 & 5 \\
  \end{bmatrix}
  \begin{bmatrix}
  a² & 2a & 8\\
  18 & 7a-4 & 10\\
  \end{bmatrix}

The first has dimensions (3,2). The second (2,3).

::

  a = np.array([
   [1,2,3],
   [4,5,6]
  ])
  a.shape == (2,3)
  b = np.array([
   [1,2,3]
  ])
  b.shape == (1,3)


Scalar operations
-----------------
Scalar operations with matrices work the same way as they do for vectors. Simply apply the scalar to every element in the matrix — add, subtract, divide, multiply, etc.

.. math::

  \begin{bmatrix}
  2 & 3 \\
  2 & 3 \\
  2 & 3 \\
  \end{bmatrix}
  +
  1
  =
  \begin{bmatrix}
  3 & 4 \\
  3 & 4 \\
  3 & 4 \\
  \end{bmatrix}

::

  # Addition
  a = np.array(
  [[1,2],
   [3,4]])
  a + 1
  [[2,3],
   [4,5]]


Elementwise operations
----------------------
In order to add, subtract, or divide two matrices they must have equal dimensions. We combine corresponding values in an elementwise fashion to produce a new matrix.

.. math::

  \begin{bmatrix}
  a & b \\
  c & d \\
  \end{bmatrix}
  +
  \begin{bmatrix}
  1 & 2\\
  3 & 4 \\
  \end{bmatrix}
  =
  \begin{bmatrix}
  a+1 & b+2\\
  c+3 & d+4 \\
  \end{bmatrix}

::

  a = np.array([
   [1,2],
   [3,4]])
  b = np.array([
   [1,2],
   [3,4]])

  a + b
  [[2, 4],
   [6, 8]]

  a — b
  [[0, 0],
   [0, 0]]


Hadamard product
----------------
Hadamard product of matrices is an elementwise operation. Values that correspond positionally are multiplied to produce a new matrix.

.. math::

  \begin{bmatrix}
  a_1 & a_2 \\
  a_3 & a_4 \\
  \end{bmatrix}
  \odot
  \begin{bmatrix}
  b_1 & b_2 \\
  b_3 & b_4 \\
  \end{bmatrix}
  =
  \begin{bmatrix}
  a_1 \cdot b_1 & a_2 \cdot b_2 \\
  a_3 \cdot b_3 & a_4 \cdot b_4 \\
  \end{bmatrix}

::

  a = np.array(
  [[2,3],
   [2,3]])
  b = np.array(
  [[3,4],
   [5,6]])

  # Uses python's multiply operator
  a * b
  [[ 6, 12],
   [10, 18]]

In numpy you can take the Hadamard product of a matrix and vector as long as their dimensions meet the requirements of broadcasting.

.. math::

  \begin{bmatrix}
  {a_1} \\
  {a_2} \\
  \end{bmatrix}
  \odot
  \begin{bmatrix}
  b_1 & b_2 \\
  b_3 & b_4 \\
  \end{bmatrix}
  =
  \begin{bmatrix}
  a_1 \cdot b_1 & a_1 \cdot b_2 \\
  a_2 \cdot b_3 & a_2 \cdot b_4 \\
  \end{bmatrix}


Matrix transpose
----------------
Neural networks frequently process weights and inputs of different sizes where the dimensions do not meet the requirements of matrix multiplication. Matrix transpose provides a way to “rotate” one of the matrices so that the operation complies with multiplication requirements and can continue. There are two steps to transpose a matrix:

  1. Rotate the matrix right 90°

  2. Reverse the order of elements in each row (e.g. [a b c] becomes [c b a])

As an example, transpose matrix M into T:

.. math::

  \begin{bmatrix}
  a & b \\
  c & d \\
  e & f \\
  \end{bmatrix}
  \quad \Rightarrow \quad
  \begin{bmatrix}
  a & c & e \\
  b & d & f \\
  \end{bmatrix}

::

  a = np.array([
     [1, 2],
     [3, 4]])

  a.T
  [[1, 3],
   [2, 4]]


Matrix multiplication
---------------------
Matrix multiplication specifies a set of rules for multiplying matrices together to produce a new matrix.

**Rules**

Not all matrices are eligible for multiplication. In addition, there is a requirement on the dimensions of the resulting matrix output. Source.

  1. The number of columns of the 1st matrix must equal the number of rows of the 2nd

  2. The product of an M x N matrix and an N x K matrix is an M x K matrix. The new matrix takes the rows of the 1st and columns of the 2nd

**Steps**

Matrix multiplication relies on dot product to multiply various combinations of rows and columns. In the image below, taken from Khan Academy’s excellent linear algebra course, each entry in Matrix C is the dot product of a row in matrix A and a column in matrix B [3]_.

.. image:: images/khan_academy_matrix_product.png
    :align: center

The operation a1 · b1 means we take the dot product of the 1st row in matrix A (1, 7) and the 1st column in matrix B (3, 5).

.. math::

  a_1 \cdot b_1 =
  \begin{bmatrix}
  1 \\
  7 \\
  \end{bmatrix}
  \cdot
  \begin{bmatrix}
  3 \\
  5 \\
  \end{bmatrix}
  = (1 \cdot 3) + (7 \cdot 5) = 38

Here’s another way to look at it:

.. math::

  \begin{bmatrix}
  a & b \\
  c & d \\
  e & f \\
  \end{bmatrix}
  \cdot
  \begin{bmatrix}
  1 & 2 \\
  3 & 4 \\
  \end{bmatrix}
  =
  \begin{bmatrix}
  1a + 3b & 2a + 4b \\
  1c + 3d & 2c + 4d \\
  1e + 3f & 2e + 4f \\
  \end{bmatrix}


Test yourself
-------------

1. What are the dimensions of the matrix product?

.. math::

  \begin{bmatrix}
  1 & 2 \\
  5 & 6 \\
  \end{bmatrix}
  \cdot
  \begin{bmatrix}
  1 & 2 & 3 \\
  5 & 6 & 7 \\
  \end{bmatrix}
  = \text{2 x 3}


2. What are the dimensions of the matrix product?

.. math::

  \begin{bmatrix}
  1 & 2 & 3 & 4 \\
  5 & 6 & 7 & 8 \\
  9 & 10 & 11 & 12 \\
  \end{bmatrix}
  \cdot
  \begin{bmatrix}
  1 & 2 \\
  5 & 6 \\
  3 & 0 \\
  2 & 1 \\
  \end{bmatrix}
  = \text{3 x 2}

3. What is the matrix product?

.. math::

  \begin{bmatrix}
  2 & 3 \\
  1 & 4 \\
  \end{bmatrix}
  \cdot
  \begin{bmatrix}
  5 & 4 \\
  3 & 5 \\
  \end{bmatrix}
  =
  \begin{bmatrix}
  19 & 23 \\
  17 & 24 \\
  \end{bmatrix}


4. What is the matrix product?}

.. math::

  \begin{bmatrix}
  3 \\
  5 \\
  \end{bmatrix}
  \cdot
  \begin{bmatrix}
  1 & 2 & 3\\
  \end{bmatrix}
  =
  \begin{bmatrix}
  3 & 6 & 9 \\
  5 & 10 & 15 \\
  \end{bmatrix}

5. What is the matrix product?

.. math::

  \begin{bmatrix}
  1 & 2 & 3\\
  \end{bmatrix}
  \cdot
  \begin{bmatrix}
  4 \\
  5 \\
  6 \\
  \end{bmatrix}
  =
  \begin{bmatrix}
  32 \\
  \end{bmatrix}



Numpy
=====

Dot product
-----------
Numpy uses the function np.dot(A,B) for both vector and matrix multiplication. It has some other interesting features and gotchas so I encourage you to read the documentation here before use.

::

  a = np.array([
   [1, 2]
   ])
  a.shape == (1,2)
  b = np.array([
   [3, 4],
   [5, 6]
   ])
  b.shape == (2,2)

  # Multiply
  mm = np.dot(a,b)
  mm == [13, 16]
  mm.shape == (1,2)


Broadcasting
------------
In numpy the dimension requirements for elementwise operations are relaxed via a mechanism called broadcasting. Two matrices are compatible if the corresponding dimensions in each matrix (rows vs rows, columns vs columns) meet the following requirements:

  1. The dimensions are equal, or

  2. One dimension is of size 1

::

  a = np.array([
   [1],
   [2]
  ])
  b = np.array([
   [3,4],
   [5,6]
  ])
  c = np.array([
   [1,2]
  ])

  # Same no. of rows
  # Different no. of columns
  # but a has one column so this works
  a * b
  [[ 3, 4],
   [10, 12]]

  # Same no. of columns
  # Different no. of rows
  # but c has one row so this works
  b * c
  [[ 3, 8],
   [5, 12]]

  # Different no. of columns
  # Different no. of rows
  # but both a and c meet the
  # size 1 requirement rule
  a + c
  [[2, 3],
   [3, 4]]


.. rubric:: Tutorials

- `Khan Academy Linear Algebra <https://medium.com/r/?url=https%3A%2F%2Fwww.khanacademy.org%2Fmath%2Flinear-algebra>`_

- `Deep Learning Book Math <https://medium.com/r/?url=http%3A%2F%2Fwww.deeplearningbook.org%2Fcontents%2Fpart_basics.html>`_

- `Andrew Ng Course Notes <https://medium.com/r/?url=https%3A%2F%2Fwww.coursera.org%2Flearn%2Fmachine-learning%2Fresources%2FJXWWS>`_

- `Linear Algebra Better Explained <https://medium.com/r/?url=https%3A%2F%2Fbetterexplained.com%2Farticles%2Flinear-algebra-guide%2F>`_

- `Understanding Matrices Intuitively <https://medium.com/r/?url=http%3A%2F%2Fblog.stata.com%2F2011%2F03%2F03%2Funderstanding-matrices-intuitively-part-1%2F>`_

- `Intro To Linear Algebra <https://medium.com/r/?url=http%3A%2F%2Fwww.holehouse.org%2Fmlclass%2F03_Linear_algebra_review.html>`_

- `Immersive Math <https://medium.com/r/?url=http%3A%2F%2Fimmersivemath.com%2Fila%2Findex.html>`_


.. rubric:: References

.. [1] http://mathinsight.org/vector_introduction
.. [2] https://en.wikipedia.org/wiki/Vector_field
.. [3] https://www.khanacademy.org/math/precalculus/precalc-matrices/properties-of-matrix-multiplication/a/properties-of-matrix-multiplication
