import matplotlib
import matplotlib.pyplot as plt
from Create_UTK_Data_files import *
from k_means_cluster import *
import sys


def cluster_avgr(np_utk_data, k_type, mu_np, num_runs):
    u, s, vh = numpy.linalg.svd(np_utk_data, full_matrices=False, compute_uv=True)
    # U *Sig*Vt = X
    v = numpy.transpose(vh)

    W = v[:, 0:k_type]
    WT = numpy.transpose(W)

    # print('shape of wt is ', W.shape)
    # grab the first two principle components for part 1-5
    W2 = v[:, 0:2]
    WT2 = numpy.transpose(W2)

    # perform the projection into k dimensional space using the principle components from above
    z_array = x_to_z_projection_pca(WT, np_utk_data, mu_np)
    z_array2 = x_to_z_projection_pca(WT2, np_utk_data, mu_np)


    num_obs = len(np_utk_data)
    #print('the number of observations: ', num_obs)
    k = k_type
    #print('k is {:d}'.format(k))
    km = k
    ko = km
    #z_array = zk
    #z_array2 = z2

    #print('m type is ', m_type)

    o_dl = list()
    k_pc = list()
    pc2 = list()
    ema = list()

    o_it = list()
    k_pc_it = list()
    pc2_it = list()
    ema_it = list()

    random_ave = num_runs

    #random_ave = 25
    #p('performing {:d} random runs'.format(random_ave))
    run_cnt = 0

    best_mean = 0;
    best_rc = list()

    try_again = False

    #for i in range(random_ave):
    while run_cnt < random_ave:
        # print('run number', run_cnt+1)
        #try_again = False
        r_c = numpy.random.choice(num_obs, km, replace=False)

        # do k means clustering using original data
        #p('doing  k means clustering using original data')
        end_mk, iter_n, bi_l, grps, vhm, dun_o = k_means_processor(np_utk_data, km, rc=r_c, verbose = False)

        # make sure we got the correct number of groups
        #if not check_grouping(grps):
        #    print('Attempting to run again')
        #    try_again=True
        #    break

        #try_again = (not check_grouping(grps))

        # do k means clustering using dimension reduced data data
        #p('do k means clustering using dimension reduced data')
        end_mk2, iter2, bi_l2, grps2, vhm2, dun_z = k_means_processor(z_array, km, rc=r_c, verbose=False)

        # make sure we got the correct number of groups
        #if not check_grouping(grps2):
        #    print('Attempting to run again')
        #    try_again=True
        #    continue

        #try_again = (not check_grouping(grps2))

        #p('do k means clustering using dimension reduced data using first 2 PC\'s')
        end_mk3, iter3, bi_l3, grps3, vhm3, dun_z2 = k_means_processor(z_array2, km, rc=r_c, verbose=False)

        # make sure we got the correct number of groups
        #if not check_grouping(grps3):
        #    print('Attempting to run again')
        #    try_again=True
        #    continue

        #try_again = (not check_grouping(grps3))

        #p('Performing Expectation Maximization.......')
        mnew, hnew, em_itera = expectation_maximization(z_array2, end_mk3, bi_l3)

        # make a grouping
        gbb = get_EM_grouping(hnew, km)
        grps4 = create_group_l(list(gbb.tolist()), km)

        # make sure we got the correct number of groups
        #if not check_grouping(grps4):
        #    print('Attempting to run again')
        #    try_again=True
        #    continue
        #    print('i should not see this')

        #try_again = (not check_grouping(grps4))

        if check_grouping(grps) and check_grouping(grps2) and check_grouping(grps3) and check_grouping(grps4):
            em_mid = min_intercluster_distance(z_array2, grps4)
            em_mxid = max_intracluster_distance(z_array2, grps4)
            dun_em = em_mid / em_mxid

            all_mean = numpy.mean(list([dun_o,dun_z, dun_z2, dun_em]))

            if all_mean > best_mean:
                best_mean = all_mean
                best_rc = r_c.copy()

            o_dl.append(dun_o)
            k_pc.append(dun_z)
            pc2.append(dun_z2)
            ema.append(dun_em)

            o_it.append(iter_n)
            k_pc_it.append(iter2)
            pc2_it.append(iter3)
            ema_it.append(em_itera)

            run_cnt = run_cnt + 1
        #else:
            #print('have to try again')

    o_dl_mu = numpy.around(numpy.array(o_dl).mean(), 2)
    k_pc_mu = numpy.around(numpy.array(k_pc).mean(), 2)
    pc2_mu = numpy.around(numpy.array(pc2).mean(),2)
    ema_mu = numpy.around(numpy.array(ema).mean(),2)

    o_it_mu = numpy.around(numpy.array(o_it).mean(), 0)
    k_it_mu = numpy.around(numpy.array(k_pc_it).mean(), 0)
    pc2_it_mu = numpy.around(numpy.array(pc2_it).mean(),0)
    ema_it_mu = numpy.around(numpy.array(ema_it).mean(),0)


    p('')
    print('---------------------------------------> For a k value of {:d}'.format(k_type))
    print('orig dun mean: {:.2f}'.format(o_dl_mu))
    print('k pc dun mean: {:.2f}'.format(k_pc_mu))
    print('1st 2 pc dun mean: {:.2f}'.format(pc2_mu))
    print('em dun: {:.2f}'.format(ema_mu))
    p('')
    print('orig iterations mean: {:.0f}'.format(o_it_mu))
    print('k pc interations mean: {:.0f}'.format(k_it_mu))
    print('1st 2 pc iterations mean: {:.0f}'.format(pc2_it_mu))
    print('em: iterations mean {:.0f}'.format(ema_it_mu))

    print('best mean', best_mean)
    print('best random choice', sorted(best_rc))
    p('---------------------------------------------------------------------------------------')
    p('')

    #if random_ave > 1:
    #    quit(0)
    return best_mean, o_dl_mu, k_pc_mu, pc2_mu, ema_mu


def multi_y_plotter(x_a, y_a, title='Multi Y  Plot', leg_a = ['red','green','purple'], x_label='X',
                    y_label='Y', best_i = -1):
    x_len = len(x_a)
    y_len = len(y_a[0])
    if x_len != y_len:
        print('x and y must be same length but x is {:d} and y is {:d}'.format(x_len, y_len))
        return -1

    fig = plt.figure(1)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    cnt = 0
    for y in y_a:
        if cnt == 0:
            symbol_type = 'r-'
        elif cnt == 1:
            symbol_type = 'g-'
        elif cnt == 2:
            symbol_type = 'm-'
        elif cnt == 3:
            symbol_type = 'b-'
        elif cnt == 4:
            symbol_type = 'y-'
        plt.plot(x_a, y, symbol_type, linewidth=2)
        cnt = cnt + 1

    plt.scatter(x_a[diff_i], (y_a[0][i]-y_a[1][i]), marker='x')

    leg = plt.legend(leg_a, loc='best',
                     borderpad=0.3, shadow=False,
                     prop=matplotlib.font_manager.FontProperties(size='medium'),
                     markerscale=0.4)

    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)

    plt.show()


data_list = load_data_files()

# un pack the data
utk_label = data_list[0]        # attrib labels for data
utk_data = data_list[1]         # data as a list of lists
np_utk_data = data_list[2]      # data as a numpy array for higher level calculations
imputated_data = data_list[3]   # avg imputated data
s_name = data_list[4]           # list of the names of the schools
head_l = data_list[5]           # list of headers?
attribs = data_list[6]          # a list where the rows are the attributes probably redundant
stats = data_list[7]            # an array of list containing various stats, unpacked below

num_obs = len(s_name)           # grab the number of observations
# print('here',num_obs)
mu_a = stats[0]                 # an array of the means of the variables/attributes
std_a = stats[1]                # an array of the stand. deviations  of the variables/attributes
min_a = stats[2]                # an array of the minimums of the variables/attributes
max_a = stats[3]                # an array of the maximums of the variables/attributes

mx_np = numpy.array(max_a)
mn_np = numpy.array(min_a)
mu_np = numpy.array(mu_a, dtype=numpy.float64)
std_np = numpy.array(std_a, dtype=numpy.float64)

np_utk_data = (np_utk_data - mu_np) / std_np
all_avg = []
orig_avg = []
kpc_avg = []
pc2_avg = []
em_avg = []

k_val_limit = 55

#num_runs = 25
num_runs = 5

for k in range(2, k_val_limit):
    all_a, orig, kpc, pc2, em = cluster_avgr(np_utk_data, k, mu_np, num_runs)
    all_avg.append(numpy.around(all_a,2))
    orig_avg.append(numpy.around(orig,2))
    kpc_avg.append(numpy.around(kpc,2))
    pc2_avg.append(numpy.around(pc2,2))
    em_avg.append(numpy.around(em,2))

y = list()
#y.append(all_avg)
y.append(orig_avg)
y.append(kpc_avg)
y.append(pc2_avg)
y.append(em_avg)
xa = list(range(2, k_val_limit))

diff_i = 0;

for i in range(len(orig_avg)):
    if(orig_avg[i] - kpc_avg[i]) < .1:
        diff_i = i
        break

print('best i is {:d}'.format(diff_i))

legend_names_a = ['Over All avg', 'Orig Avg', 'K PC\'s avg']
legend_names_b = ['Over All avg', 'Orig Avg', 'K PC\'s avg', '2 PC\'s', 'EM']
legend_names_c = ['Orig Avg', 'K PC\'s avg', '2 PC\'s', 'EM']


multi_y_plotter(xa, y, title='K vs Dunn index', leg_a=legend_names_c, x_label='K value',
                y_label='Dunn index average', diff_i=diff_i)
