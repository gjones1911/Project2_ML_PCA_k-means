import numpy


# principle component analysis
def x_to_z_projection_pca(WT, x_data, mu_array):

    z_array = list()

    for row in x_data:
        c_x = row-mu_array
        z_array.append(numpy.dot(WT, c_x))

    Z = numpy.array(z_array, dtype=numpy.float)

    return Z


def x_to_z_projection_pca2(x_data, mu_array, k):
    # use numpy to perform spectral vector decomposition for PCA
    u, s, vh = numpy.linalg.svd(x_data, full_matrices=True, compute_uv=True)

    vt = numpy.transpose(vh)

    # grab the first k principle components
    W = vt[:, 0:k]
    WT = numpy.transpose(W)

    # grab the first two principle components
    W2 = vt[:, 0:2]
    WT2 = numpy.transpose(W2)

    z_array = list()
    z2_array = list()

    for row in x_data:
        c_x = row-mu_array
        z_array.append(numpy.dot(WT, c_x))
        z2_array.append(numpy.dot(WT2, c_x))

    Z = numpy.array(z_array, dtype=numpy.float)
    Z2 = numpy.array(z2_array, dtype=numpy.float)

    return Z, Z2
