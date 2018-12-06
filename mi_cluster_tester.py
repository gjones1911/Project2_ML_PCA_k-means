from Create_UTK_Data_files import *
from k_means_cluster import *
import sys


def cluster_tester(np_utk_data, k_type, zk, z2, num_runs):

    num_obs = len(np_utk_data)
    print('the number of observations: ', num_obs)
    k = k_type
    print('k is {:d}'.format(k))
    km = k
    ko = km
    z_array = zk
    z_array2 = z2

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
    p('performing {:d} random runs'.format(random_ave))
    run_cnt = 0

    best_mean = 0;
    best_rc = list()

    try_again = False

    #for i in range(random_ave):
    while run_cnt < random_ave:
        print('run number', run_cnt+1)
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
        else:
            print('have to try again')

    o_dl_mu = numpy.around(numpy.array(o_dl).mean(), 2)
    k_pc_mu = numpy.around(numpy.array(k_pc).mean(), 2)
    pc2_mu = numpy.around(numpy.array(pc2).mean(),2)
    ema_mu = numpy.around(numpy.array(ema).mean(),2)

    o_it_mu = numpy.around(numpy.array(o_it).mean(), 0)
    k_it_mu = numpy.around(numpy.array(k_pc_it).mean(), 0)
    pc2_it_mu = numpy.around(numpy.array(pc2_it).mean(),0)
    ema_it_mu = numpy.around(numpy.array(ema_it).mean(),0)


    p('')
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

    #if random_ave > 1:
    #    quit(0)
    return sorted(best_rc), best_mean,
    #return best_rc, best_mean


