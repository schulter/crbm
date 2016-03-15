from theano.gof.op import get_debug_values
import theano.tensor as T


def max_pool(z, pool_shape, top_down=None, theano_rng=None):
    """
    Probabilistic max-pooling
    Parameters
    ----------
    z : theano 4-tensor
        a theano 4-tensor representing input from below
    pool_shape : tuple
        tuple of ints. the shape of regions to be pooled
    top_down : theano 4-tensor, optional
        a theano 4-tensor representing input from above
        if None, assumes top-down input is 0
    theano_rng : MRG_RandomStreams, optional
        Used for random numbers for sampling
    Returns
    -------
    p : theano 4-tensor
        the expected value of the pooling layer p
    h : theano 4-tensor
        the expected value of the detector layer h
    p_samples : theano 4-tensor, only returned if theano_rng is not None
        samples of the pooling layer
    h_samples : theano 4-tensor, only returned if theano_rng is not None
        samples of the detector layer
    Notes
    ------
    all 4-tensors are formatted with axes ('b', 'c', 0, 1).
    This is for maximum speed when using theano's conv2d
    to generate z and top_down, or when using it to infer conditionals of
    other layers using the return values.
    Detailed description:
    Suppose you have a variable h that lives in a Conv2DSpace h_space and
    you want to pool it down to a variable p that lives in a smaller
    Conv2DSpace p.
    This function does that, using non-overlapping pools.
    Specifically, consider one channel of h. h must have a height that is a
    multiple of pool_shape[0] and a width that is a multiple of pool_shape[1].
    A channel of h can thus be broken down into non-overlapping rectangles
    of shape pool_shape.
    Now consider one rectangular pooled region within one channel of h.
    I now use 'h' to refer just to this rectangle, and 'p' to refer to
    just the one pooling unit associated with that rectangle.
    We assume that the space that h and p live in is constrained such
    that h and p are both binary and p = max(h). To reduce the state-space
    in order to make probabilistic computations cheaper we also
    constrain sum(h) <= 1.
    Suppose h contains k different units. Suppose that the only term
    in the model's energy function involving h is -(z*h).sum()
    (elemwise multiplication) and the only term in
    the model's energy function involving p is -(top_down*p).sum().
    Then P(h[i] = 1) = softmax( [ z[1], z[2], ..., z[k], -top_down] )[i]
    and P(p = 1) = 1-softmax( [z[1], z[2], ..., z[k], -top_down])[k]
    This variation of the function assumes that z, top_down, and all
    return values use Conv2D axes ('b', 'c', 0, 1).
    This variation of the function implements the softmax using a
    theano graph of exp, maximum, sub, and div operations.
    """

    z_name = z.name
    if z_name is None:
        z_name = 'anon_z'

    batch_size, ch, zr, zc = z.shape

    r, c = pool_shape

    zpart = []

    mx = None

    if top_down is None:
        t = 0.
    else:
        t = - top_down
        t.name = 'neg_top_down'

    for i in xrange(r):
        zpart.append([])
        for j in xrange(c):
            cur_part = z[:, :, i:zr:r, j:zc:c]
            if z_name is not None:
                cur_part.name = z_name + '[%d,%d]' % (i, j)
            zpart[i].append(cur_part)
            if mx is None:
                mx = T.maximum(t, cur_part)
                if cur_part.name is not None:
                    mx.name = 'max(-top_down,' + cur_part.name + ')'
            else:
                max_name = None
                if cur_part.name is not None:
                    mx_name = 'max(' + cur_part.name + ',' + mx.name + ')'
                mx = T.maximum(mx, cur_part)
                mx.name = mx_name
    mx.name = 'local_max(' + z_name + ')'

    pt = []

    for i in xrange(r):
        pt.append([])
        for j in xrange(c):
            z_ij = zpart[i][j]
            safe = z_ij - mx
            safe.name = 'safe_z(%s)' % z_ij.name
            cur_pt = T.exp(safe)
            cur_pt.name = 'pt(%s)' % z_ij.name
            pt[-1].append(cur_pt)

    off_pt = T.exp(t - mx)
    off_pt.name = 'p_tilde_off(%s)' % z_name
    denom = off_pt

    for i in xrange(r):
        for j in xrange(c):
            denom = denom + pt[i][j]
    denom.name = 'denom(%s)' % z_name

    off_prob = off_pt / denom
    p = 1. - off_prob
    p.name = 'p(%s)' % z_name

    hpart = []
    for i in xrange(r):
        hpart.append([pt_ij / denom for pt_ij in pt[i]])

    h = T.alloc(0., batch_size, ch, zr, zc)

    for i in xrange(r):
        for j in xrange(c):
            h.name = 'h_interm'
            h = T.set_subtensor(h[:, :, i:zr:r, j:zc:c], hpart[i][j])

    h.name = 'h(%s)' % z_name

    if theano_rng is None:
        return p, h

    else:
        events = []
        for i in xrange(r):
            for j in xrange(c):
                events.append(hpart[i][j])
        events.append(off_prob)

        events = [event.dimshuffle(0, 1, 2, 3, 'x') for event in events]

        events = tuple(events)

        stacked_events = T.concatenate(events, axis=4)

        rows = zr // pool_shape[0]
        cols = zc // pool_shape[1]
        outcomes = pool_shape[0] * pool_shape[1] + 1
        assert stacked_events.ndim == 5
        for se, bs, r, c, chv in get_debug_values(stacked_events, batch_size,
                                                  rows, cols, ch):
            assert se.shape[0] == bs
            assert se.shape[1] == r
            assert se.shape[2] == c
            assert se.shape[3] == chv
            assert se.shape[4] == outcomes
        reshaped_events = stacked_events.reshape((
            batch_size * rows * cols * ch, outcomes))

        multinomial = theano_rng.multinomial(pvals=reshaped_events,
                                             dtype=p.dtype)

        reshaped_multinomial = multinomial.reshape((batch_size, ch, rows,
                                                    cols, outcomes))

        h_sample = T.alloc(0., batch_size, ch, zr, zc)

        idx = 0
        for i in xrange(r):
            for j in xrange(c):
                h_sample = T.set_subtensor(h_sample[:, :, i:zr:r, j:zc:c],
                                           reshaped_multinomial[:, :, :, :,
                                           idx])
                idx += 1

        p_sample = 1 - reshaped_multinomial[:, :, :, :, -1]

        return p, h, p_sample, h_sample


def max_pool_us(z, pool_shape, theano_rng=None):
    # Firstly, get some shape information
    N_batch, K, _, N_h = z.shape
    vertical_unit_num = K // pool_shape[0]
    horizontal_unit_num = N_h // pool_shape[1]
    
    # STEP 1: calculate softmax probabilites for the units
    # reshape in a way such that units that will be pooled together
    # are close to each other
    R = z.reshape((N_batch,
                   vertical_unit_num,
                   1 * pool_shape[0],
                   horizontal_unit_num,
                   pool_shape[1])).astype(z.dtype)
    
    # 2. exponentiate everything
    R = T.exp(R)

    # calculate denominators for softmax
    denom = R.sum(axis=4, keepdims=True).sum(axis=2, keepdims=True).astype(R.dtype)
    denom_small = R.sum(axis=4).sum(axis=2)
    denom_small = denom_small.reshape((denom_small.shape[0], denom_small.shape[1], denom_small.shape[2], 1))
    
    # now, divide our reshaped matrix by denominators to get the probability matrix
    # remember: softmax_i = exp(x_i) / (1 + sum_over_j exp(x_j))
    div = R / (denom + 1)
    probabilities_H = div.reshape(z.shape)

    # STEP 2: sample from the softmax distribution calculated earlier
    # for sampling, get the units to have one dimension only
    div_s = div.dimshuffle(0, 1, 3, 2, 4)
    div_s = div_s.reshape((div_s.shape[0], div_s.shape[1], div_s.shape[2], -1))

    # now, add 1 / (denom + 1) to the last dim to represent no activation
    no_activation = T.ones((div_s.shape[0], div_s.shape[1], div_s.shape[2], 1)) / (1 + denom_small)
    with_no_act = T.concatenate((div_s, no_activation), axis=3)
    probabilities_P = div.sum(axis=(2,4), keepdims=True)

    # The MRG random stream only supports 2D matrices -> reshape to that
    multinomial_drawing_matrix = with_no_act.reshape((with_no_act.shape[0]*with_no_act.shape[1]*with_no_act.shape[2],
                                                      with_no_act.shape[3]))

    # do the drawing
    multinomial = theano_rng.multinomial(pvals=multinomial_drawing_matrix, dtype=multinomial_drawing_matrix.dtype)
    multinomial = multinomial[:,:-1] # remove the no-activation unit

    # reshape back to original shapes
    sample_H = multinomial.reshape((N_batch, vertical_unit_num, horizontal_unit_num, pool_shape[0], pool_shape[1]))
    sample_H = sample_H.dimshuffle(0, 1, 3, 2, 4)
    sample_H = sample_H.reshape(z.shape)
    
    # get P (probabilites and sample reduced (one number per unit)

    return T.zeros(z.shape), probabilities_H, T.zeros(z.shape), sample_H
