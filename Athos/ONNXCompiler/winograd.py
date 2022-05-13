import numpy as np
import pickle as pkl


def get_modified_kernel(
    kernel, output_dim, constants, indices, numerators, denominators
):
    mul_terms = []
    for consts, inds, den in zip(constants, indices, denominators):
        summ = 0
        for c, i in zip(consts, inds):
            summ += c * kernel[i]
        mul_terms += [summ / den]

    return np.array(mul_terms).reshape(output_dim, output_dim)

def get_modified_filter(filter, transform, n):
    filter_m = np.random.rand(filter.shape[0], filter.shape[1], n, n)

    constants = transform["constants"]
    indices = transform["indices"]
    numerators = transform["numerators"]
    denominators = transform["denominators"]

    for i in range(filter.shape[0]):
        for j in range(filter.shape[1]):
            filter_m[i][j] = get_modified_kernel(
                filter[i][j], n, constants, indices, numerators, denominators
            )

    return filter_m

if __name__ == "__main__":
    pass
