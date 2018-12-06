from ML_Visualizations import *
from Process_CSV_to_Json import get_basic_stats
from DimensionReduction import x_to_z_projection_pca
from collections import Counter

# -----------------------------------------Utility functions(make things easy---------------------------------------
# cheap print function
def p(str):
    print(str)
    return


# random number generagor
def rng(start, stop, step):
    return numpy.random.choice(range(start, stop+step, step), 1)

# ------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- Reference Vector Functions ----------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# makes a random mk vector to be used for initial clustering
def make_rand_m(data_array, k, rc=None):

    row_col = data_array.shape

    rows = row_col[0]
    cols = row_col[1]

    if rc is None:
        rand_chc = numpy.random.choice(rows, k, replace=False)
    else:
        #print(rc)
        rand_chc = rc

    #rand_chc = numpy.array([55,2,42,25,8,22,21,52,12])

    ret_l = list()

    for inst in rand_chc:
        ret_l.append(data_array[inst])

    #mk = numpy.stack(ret_l)
    mk = numpy.array(ret_l, dtype=numpy.float)

    return rand_chc, mk


# uses r_c to pick out vectors in x to make a set of reference vectors
def make_given_m(x, r_c):
    ret_l = list()
    for inst in r_c:
        ret_l.append(x[inst])
        # mk = numpy.stack(ret_l)
    mk = numpy.array(ret_l, dtype=numpy.float)
    return mk


# makes  a mean based set of referecne vectors
def make_mean_mod_m(x, k):

    mu = x.mean(axis=0, dtype=numpy.float64)
    mn = x.min(axis=0)
    mx = x.max(axis=0)

    ret_l = list()

    div = list([60])

    for i in range(k):
        div.append(div[-1]+5)

    for i in range(k):
        #ret_l.append(mu + mn/rng(50, 75, 5))
        #ret_l.append(mu + mn/div[i])
        #ret_l.append(mu + ((-1**i)*(mn/rng(0, 1, .05))))
        ret_l.append(mu + ((1**i)*(mn*.5)*i/10))

    return numpy.stack(ret_l)


def get_mid_p(glist, x):
    r_c = x.shape
    c = r_c[1]
    #xl = numpy.array([0]*c, dtype=numpy.float)
    yl = 0
    xlist = list()
    for i in glist:
        xlist.append(list(x[i].tolist()))
    xl = numpy.stack(xlist)
    x_stat = get_basic_stats(xl)
    return x_stat[0]


def make_g_m(x):
    g1 = [22]
    g2 = [3]
    g3 = [25,26]
    g4 =[19,14,27,12]
    g5 =[5,17,24,4,23,18,2]
    g6 = [44,6,38,40,7,46,1,11,12]
    g7 = [34,29,0,10,33,53,37,47,9,39,36,31,28,41,45,20,55]
    g8 = [32,56,43,49,51,35,21,48]
    g9 = [15,16,52,8,30,54,13,50]

    size = len(g1) + len(g2) + len(g3) + len(g4) + len(g5) + len(g6) + len(g7) + len(g8) + len(g9)
    print('the size is {:d}'.format(size))
    gtot = [g1, g2, g3, g4, g5, g6, g7, g8, g9]

    m_all = list()

    for g in gtot:
        m_all.append(get_mid_p(g, x))

    return numpy.array(m_all, dtype=numpy.float)

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# ------------------------------------- k means functions ------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------


def project_mid_points(end_mk, v, k):
    vt = numpy.transpose(v)
    Wm = vt[:, 0:k]
    WTm = numpy.transpose(Wm)
    m_stats = get_basic_stats(end_mk)
    mid_points = x_to_z_projection_pca(WTm, end_mk, numpy.array(m_stats[0], dtype=numpy.float))
    return mid_points


# calculate the minimum intercluster distance
# calculate the minimum distance between points in different
# clusters
def min_intercluster_distance(x, gs):

    mid = 100**5

    # get each group
    for g in range(len(gs)-1):

        # for every entry in this group
        # each entry is a point in the group
        for entry in gs[g]:
            #get a entry in group i
            x1 = x[entry]
            # look at every other group and get the distance between the entries in that
            # group and group g and save the min
            for g2 in range(g+1, len(gs)):
                for entryj in gs[g2]:
                    x2 = x[entryj]
                    dis = numpy.linalg.norm(x1-x2)
                    # keep track of the minimum distance
                    if dis < mid:
                        mid = dis
    return mid


# returns maximun intracluster distance e.g.
# the largest distance between points in same clusters
def max_intracluster_distance(x, gs):

    maxid = 0

    # for each group
    for row in gs:
        #go through  current groups entries except last
        for i in range(len(row)-1):
            # get the current rows ith idx
            cidx = row[i]
            # go through all other entries in the current row/group
            for j in range(i+1, len(row)):
                #get the next or jth idx in row/group
                nidx = row[j]
                # get distance between this two points
                #dis = numpy.linalg.norm(x[i] - x[j])
                dis = numpy.linalg.norm(x[cidx] - x[nidx])
                # keep track of largest distance
                if dis > maxid:
                    maxid = dis
    return maxid


# calculate a label matrix
def calculate_bi(x, m):

    # N x d
    r_c = x.shape

    # K x d
    mr_mc = m.shape

    rows = r_c[0]

    bi_list = list()

    # go through every observation
    for i in range(0, rows):
        bi = [0]*mr_mc[0]
        min_l = list()
        mmn = 10000**6
        jsave = list()
        # go through every reverence vector
        # calculating the difference between the
        # current observation vector and the jth
        # reverence vector. store the minimum length
        for j in range(0, mr_mc[0]):
            dif = x[i] - m[j]
            norm = numpy.linalg.norm(dif)
            if mmn > norm:
                mmn = norm
                jsave.clear()
                jsave.append(j)
            elif mmn == norm:
                jsave.append(j)

        for indx in jsave:
            bi[indx] = 1
        '''
        for m_row in m:
            #dif = x[i]-m[j]
            dif = x[i]-m_row
            norm = numpy.linalg.norm(dif)
            min_l.append(norm)

        minimum = min(min_l)
        #print('the length of min_l is {:d}'.format(len(min_l)))
        #print('the length of bi is {:d}'.format(len(bi)))
        #p('The min was {:f}'.format(minimum))
        # add a 1 to the column to signify where this observation (row)
        # is in group id
        found = False
        for id in range(len(min_l)):
            if min_l[id] == minimum:
                bi[id] = 1
                found = True
        if not found:
            p('')
            p('')
            p('I DID NOT FIND THE MIN')
            p('')
            p('')
        '''

        # print(bi)
        bi_list.append(bi)
    # create_group_l(bi_list, len(bi))

    return numpy.array(bi_list, dtype=numpy.float)


# create new reference vectors
def get_new_m(x, m, bi):

    new_m = m.tolist()

    # go through the k rows of m
    for i in range(len(m)):
        l = [0]*len(x[0])
        sm = numpy.array(l, dtype=numpy.float64)
        bs = 0
        in_g = 0
        # go through t rows  of x
        for row in range(len(x)):
            # print('bval is {:f}'.format(bi[row][i]))
            # look at label array at b[t][i]
            # and if it is a one some x[row] with the rest
            bval = bi[row][i]
            if bval == 1:
                in_g += 1
                sm += x[row]
            bs += bval
        # print('------------------------------------------------------------------sm is now {:d}'.format(in_g))
        #if in_g == 0:
        #    new_m[i] = m[i]
        #    new_m[i] = [0]*len(x[0])
        #else:
        if bs == 0:
            bs = .00001

        new_m[i] = sm/bs

    np_new_m = numpy.array(new_m, dtype=numpy.float)

    # save how much the new and old differ
    dif_m = abs(np_new_m - m)
    return np_new_m, dif_m


# def k_means_clustering(x, k, mu_a, min_a, max_a):
def k_means_clustering(x, k, init_m=[], m_init_type=0, mu_a=numpy.array([]),
                       min_a=numpy.array([]), max_a=numpy.array([]), rc=None):
    if len(init_m) == 0:
        #p('made it')
        if m_init_type == 0:
            #p('initialize m to random k elements of x')

            if rc is None:
                r_c, mk = make_rand_m(x, k)
            else:
                #p('trying to use rc')
                r_c, mk = make_rand_m(x, k, rc=rc)
        elif m_init_type == 1:
            r_c = []
            mk = make_g_m(x)
        elif m_init_type == 2:
            p('initialize m to modified mean k elements of x')
            r_c = []
            mk = make_mean_mod_m(x,k)
    else:
        mk = init_m

    avg_dif = 10000

    iter = 0

    #while abs(avg_dif) > 0:
    while numpy.around(abs(avg_dif),3) > .001:
        bi_list = calculate_bi(x, mk)
        mk, dif_m = get_new_m(x, mk[:, :], bi_list)
        avg_dif = numpy.sum(dif_m)
        iter += 1

    return mk, iter, bi_list


def k_means_processor(x, k, m_init_type=0, rc=None, verbose=True):
    found = False

    while not found:
        if m_init_type != 0:
            end_mk, iter_n, bi_l = k_means_clustering(x, k, m_init_type=m_init_type)
        elif rc is None:
            end_mk, iter_n, bi_l = k_means_clustering(x, k)
        else:
            if verbose:
                print('attempting to use passed rc')
            end_mk, iter_n, bi_l = k_means_clustering(x, k, rc=rc)
           # p('')
           # p('')
        grps = create_group_l(list(bi_l.tolist()), k)

        # make sure I found the number of groups I wanted
        found = check_grouping(grps)
        #if not found:
        #    rc = None
        found = True


    um, sm, vhm = numpy.linalg.svd(end_mk, full_matrices=True, compute_uv=True)

    mid = min_intercluster_distance(x, grps)
    mxid = max_intracluster_distance(x, grps)
    if verbose:
        p('The number of interations: {:d}'.format(iter_n))
        p('The min intercluster distance is : {:f}'.format(mid))
        p('The max intracluster distance is : {:f}'.format(mxid))

    dun_o = mid / mxid
    if verbose:
        p('the dun index is for k-means of original data is: {:f}'.format(dun_o))

    return end_mk, iter_n, bi_l, grps, vhm, dun_o



# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# ------------------------- Dimensional conversion methods ----------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------


def x_reduced(u, s, vt):

    us = numpy.dot(u,s)

    return numpy.dot(us, vt)


def reduce_x(W, z, mu):

    l = []
    cnt = 0
    x_array = list()

    for row in z:
        res = numpy.dot(W, row)
        x_array.append(res + mu)

    x = numpy.array(x_array, dtype=numpy.float)

    return x

# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -------------------------------------- Estimation Maximization methods ------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------


def get_si(x, m, h):

    s = list()
    S = numpy.array([0]*len(m), dtype=numpy.float)
    for i in range(len(m)):
        st = 0
        sb = 0
        for t in range(len(x)):
            st += h[t][i]*(numpy.dot((x[t]-m[i]), numpy.transpose(x[t]-m[i])))
            sb += h[t][i]
        if sb == 0:
            S[i] = 0
        else:
            S[i] = st/sb

    return numpy.array(S, dtype=numpy.float)


def calculate_pi_i(h, N):
    pi_i = list()

    r_c = h.shape

    for i in range(r_c[1]):
        sm = 0
        for t in range(N):
            sm += h[t][i]
        pi_i.append(sm/N)
    return numpy.array(pi_i, dtype=numpy.float)


def e_step(S, x, m, pi_i):
    ht = list()
    for t in range(len(x)):
        sb = 0
        hi = list()
        for j in range(len(m)):
            dif = x[t] - m[j]
            s = S[j]
            if s == 0:
                s = .0000001
            a1 = -.5 * numpy.transpose(dif)
            b1 = (1 / s) * dif
            sb += pi_i[j] * (abs(s)**(-.5)) * numpy.exp( numpy.dot(a1, b1) )
        for i in range(len(m)):
            s = S[i]
            if s == 0:
                s = .0000001
            dif = x[t] - m[i]
            a1 = -.5*numpy.transpose(dif)
            b1 = (1/s) * dif
            val = pi_i[i] * (abs(s)**(-.5)) * numpy.exp(numpy.dot(a1, b1) )
            hi.append(val/sb)
        ht.append(hi)

    return numpy.array(ht, dtype=numpy.float)


def calculate_mi(h, x):
    m = list()

    for i in range(len(h[0])):

        sm = numpy.array([], dtype=numpy.float)

        tp = numpy.array([0] * len(x[0]), dtype=numpy.float)

        bt = 0

        for t in range(len(x)):

            #print(h[t][i])
            tp += h[t][i] * x[t]
            bt += h[t][i]

        if bt == 0:
            bt = .00001

        m.append(tp / bt)

    return numpy.array(m, dtype=numpy.float)


def expectation_maximization(x, m, h):
    # estimate S
    S = get_si(x, m, h)

    #calculate pi_i
    pi_i = calculate_pi_i(h, len(x))

    hi = e_step(S, x, m, pi_i)

    mi = calculate_mi(hi, x)
    dif = 0
    ret_d = 100000
    itera = 0

    while abs(ret_d) > 1:
        mold = numpy.array(list(mi.tolist()))

        dif_old = dif

        # calculate new si
        s = get_si(x, mi, hi)

        # calculate pi_i
        pi_i = calculate_pi_i(hi, len(x))

        # perform e step
        hi = e_step(s, x, mi, pi_i)

        # perform m step
        mi = calculate_mi(hi, x)

        # calculate difference between the old and new reference
        # vectors
        dif = numpy.mean(mold - mi, dtype=numpy.float)

        ret_d = dif.cumsum()[-1]

        #ret_d = numpy.around(abs(dif_old - dif), 1)
        #ret_d = numpy.around(abs(numpy.mean(mold - mi)), 1)
        itera +=1
        #print('The dif is now {:f}'.format(ret_d))

    return mi, hi, itera

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# -------------------------------------- Grouping and labeling functions -----------------------------------------


# used to make sure we got all the groups we wanted
def check_grouping(groups):
    cnt = 0
    for group in groups:
        if len(group) == 0:
            p('Wrong number of groups found at {:d}'.format(cnt))
            return False
        cnt += 1
    return True


# makes a list where each row is a group and the entries in the list
# at this row are what observations belong to that group
def create_group_l(b, k):
    group_list = list()
    for i in range(k):
        group_list.append(list())
    for obs in range(len(b)):
        # look in current b row and find what group this observation
        # belongs to
        for n in range(len(b[obs])):
            if b[obs][n] == 1:
                #gnum = b[obs].index(1)
                group_list[n].append(obs)
        #p('the group number is {:d}'.format(gnum))
        #group_list[gnum].append(obs)
    #p('')
    #p('')
    #p('')
    #p('')
    #p('')
    for i in range(len(group_list)):
        if len(group_list[i]) == 0:
            p('No one in group {:d}'.format(i))

    return group_list


# makes a list where each row is a group and the entries in the list
# at this row are what observations belong to that group
def create_group_names(group, names):
    name_list = list()
    for row in group:
        # look in current b row and find what group this observation
        # belongs to
        l = list()
        for idx in row:
            #gnum = b[obs].index(1)

            l.append(str(idx) + ': ' + names[idx].strip('\n') )

        name_list.append(l)

    return name_list


def get_EM_grouping(h, k):
    hl = h.tolist()
    g = list()
    #for i in range(k):
    #    l = list([0]*k)
    #    g.append(l)

    for row in range(len(hl)):
        mx = max(hl[row])
        l_count = Counter(hl[row])

        if l_count[mx] > 1:
            print('there are {:d}'.format(l_count[mx]))

        l = list([0]*k)
        g.append(l)

        g[-1][hl[row].index(mx)] = 1

    return numpy.array(g, dtype=numpy.float)


def show_grouping(grouping):

    r = 1
    for row in grouping:
        print('Group {:d}: '.format(r))
        p(row)
        p('')
        r += 1
    return


def make_label_list(size):
    groups = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'k', 'L', 'M', 'N', 'O', 'P',
               'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    ret_l = [''] * size

    r = list

    for i in range(size):

        ret_l[i] = ret_l[i%len(groups)] + groups[i%len(groups)]

    return ret_l

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

