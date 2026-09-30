# Verifier scaling design (Problem C, N up to ~50,000; Problem A, n up to a few hundred)

**Rigor requirement.** A certified value must be an exact upper bound, never a floating-point FFT estimate.

## Problem C: exact autoconvolution maximum
For a step function with N equal pieces on [-1/4, 1/4] and heights a_i >= 0 the certified quantity is
`R = 2N * max_m c_m / (sum_i a_i)^2`, `c_m = sum_{i+j=m} a_i a_j`.
(Why this is the right quantity: f*f is piecewise linear with vertices at t_m = -1/2 + (m+1)h, h = 1/(2N), and vertex values h*c_m,
so max(f*f) = h*max_m c_m and int f = h*sum a; hence max(f*f)/(int f)^2 = 2N max c / (sum a)^2, and this is an upper bound on C_1.)

The certificate holds the *decimal text* of every height. The verifier converts to exact integers `A_i = a_i * D` (D = lcm of the
denominators, i.e. a power of 10), so `c_m` is an exact integer and R an exact rational (`Fraction`), reported to 30 digits.

Two independent exact algorithms (`verify/ac1.py`), one per backend, cross-checked in the tests and at certification time:

| backend | algorithm | cost at N = 50,000 |
|---|---|---|
| `fraction` (exact rational) | small N (<= 200): direct O(N^2) integer loop; large N: **Kronecker substitution** - pack all A_i into one big Python integer with slot width > bit_length(N * max(A)^2), square it with exact big-int multiplication, unpack the slots (no carries between slots by construction) | ~1.4 s |
| `mpmath` (60-digit) | small N: mpf O(N^2) convolution; large N: **int64 limb convolution** - split A_i into 20-bit limbs, `np.convolve` on int64 (every partial sum < 2^63 for N <= 2^22, asserted), recombine limbs with Python ints; final division in 60-digit mpmath | ~2 s at N = 20,000 |

No floating-point FFT/NTT is used anywhere in the verifier, so there is nothing to round. `tests/test_verify.py` checks that direct,
Kronecker and limb algorithms return identical lists on random inputs, all-maximal inputs (slot-carry adversary), single spikes, and
that both backends agree on N = 700 certificates; gzip-compressed certificates (`*.json.gz`) are read transparently.

## Problem A: all-pairs, not neighbour lists
The verifier checks every pair (i < j) exactly (Fractions) and in 60-digit mpmath - it never uses neighbour lists. Neighbour lists exist only
in the *search* (`search/large.py`); their exactness is tested (`tests/test_large.py`): the LP restricted to pairs with gap < (2*sqrt(2)+2)*Delta
cannot lose a violated pair inside a trust region of size Delta, and a brute-force all-pairs check after each step confirms feasibility. A test
also plants an overlap between circles with far-apart indices in an n = 120 certificate and confirms both backends reject it.
The polish (`search/polish.py::polish_mixed`) evaluates residuals in 60-digit mpmath (float64 only for the linear solve), then a full
high-precision all-relevant-pairs check and radius shrink precede the certificate; the verifier re-checks everything independently.
