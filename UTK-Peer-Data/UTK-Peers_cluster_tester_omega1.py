from Create_UTK_Data_files import *
from k_means_cluster import *
import sys


# --------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- PART 1 --------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------- Part1-1 -------------------------------------------------------------
# Load  a data matrix out of given data
# loaded needed data files for utk data
# default loads the utk data
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
mu_np = numpy.array(mu_a)
std_np = numpy.array(std_a)

data_type = 0
while data_type != 1 and data_type != 2 and data_type != 3:
    data_type = int(input('Do you want to use 1) normalized or 2) unnormalized? '))
    if data_type != 1 and data_type != 2:
        print('You must pick 1 or 2')


k_type = int(input('Do you want to use k as: -1) --the k chosen by pov\n'
                   '                         -2) --the k chosen by elbow of scree\n'
                   '                         any int >1 --some k chosen by you '))
if k_type == -2 or k_type > 1:
    m_type = 3
else:
    m_type = -1
    while 0 > m_type:
        if k_type == -1:
            m_type = int(input('What type of initial m do you want to use: 1) chosen m k size 5\n'
                               '                                           2) random m '))

        elif k_type == -2:
            m_type = int(input('What type of initial m do you want to use: 1) chosen m k size 2\n'
                               '                                           2) random m '))
        if m_type > 2:
            print('Choice must be 1,2, or 3')
            m_type = -1
        elif m_type == 2:
            m_type = 3
        elif m_type == 1:
            if data_type == 1:
                m_type = 1
            elif data_type == 2:
                m_type = 2


print('mtype', m_type)

data_type = 3

if data_type == 1:
    np_utk_data = (np_utk_data-mn_np)/(mx_np-mn_np)
elif data_type == 3:
    np_utk_data = (np_utk_data - mu_np) / std_np

#np_utk_data = (np_utk_data - mu_np) / std_np

# ------------------------------------- Part1-2  -----------------------------------------------------------------------
# use numpy to perform spectral vector decomposition for PCA
u, s, vh = numpy.linalg.svd(np_utk_data, full_matrices=False, compute_uv=True)
# U *Sig*Vt = X
v = numpy.transpose(vh)

# -----------------------------------------------------------------------------------------------------------------
# use the prop of variance graphing function to get a good estimate of k

# ------------------------------------- Part1-3 -------------------------------------------------------------------
# -------------------------------------plot a scree graph of singular values and % of variance using (s)
# -------------------------------------use this to guess a good value for k
kpov = make_prop_o_var_plot(s, len(s), show_it=False, last_plot=False)
ks = make_scree_plot_usv(s, len(s), show_it=False, last=False, k_val=kpov)
k = kpov

if k_type == -1:
    k = kpov
    print('using the POV estimated k')
elif k_type == -2:
    k = ks
    print('using the scree estimated k')
else:
    print('using the user given k')
    k = k_type

# p('The original k:')
# p('The k for original data is {:d}'.format(k))
km = k
# ------------------------------------------------------------------------------------------------------------------

# -------------------------------------------- Part1-4 ----------------------------------------------------------------
ko = km
print('The k value is ', k)
# grab the first k principle components vectors
W = v[:, 0:k]
WT = numpy.transpose(W)
print('shape of wt is ', W.shape)
# grab the first two principle components for part 1-5
W2 = v[:, 0:2]
WT2 = numpy.transpose(W2)
# perform the projection using the principle components from above
z_array = x_to_z_projection_pca(WT, np_utk_data, numpy.array(mu_a, dtype=numpy.float))
z_array2 = x_to_z_projection_pca(WT2, np_utk_data, numpy.array(mu_a, dtype=numpy.float))

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Part1-5 ---------------------------------------------------------------

# set up the scatter plot of the data
if data_type == 1:
    z_title = 'The first 2 PCs for the normalized data'
else:
    z_title = 'The first 2 PCs for the transformed data into {:d} dimensional space'.format(k)

z_scatter_plot(z_array, s_name, title=z_title, last=False, show_it=False)
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------- PART 2 ---------------------------------------------------------------


# --------------------------------------------- Part 2-6,7,8,9,10,11,12 -----------------------------------------------

#        make a random set of k initial reference variables

print('m type is ', m_type)

o_dl = list()
k_pc = list()
pc2 = list()
ema = list()

o_it = list()
k_pc_it = list()
pc2_it = list()
ema_it = list()

if k_type > 0:
    random_ave = 25
else:
    random_ave = 1

#random_ave = 25
p('performing {:d} random runs'.format(random_ave))
run_cnt = 1

best_mean = 0;
best_rc = list()

for i in range(random_ave):
    if m_type == 2:
        #r_c = list([3, 15])
        #r_c = list([22, 55])
        #r_c = list([22, 14])
        r_c = list([22, 19])
    elif m_type == 1:
        #r_c = list([22, 14, 2, 53, 31])
        r_c = list([22, 14, 2, 53, 31])
    elif m_type >= 3:
        r_c = numpy.random.choice(num_obs, km, replace=False)

    #        Unnormalized
        #           k = 3
        #r_c = list([3,25,0]) # the one i like for unnorm k=4
        #           k = 7
        #r_c = list([3,19,4,8,51,44,55])
        #           k = 6
        #r_c = list([3,19,4,8,51,0]) #unnorm
        #           k = 5
        #r_c = list([2,5,16,50,29,39]) #unnorm
        #           k = 4
        #r_c = list([2,5,24,39]) #unnorm

        #        Normalized
        #           k = 4
        #r_c = [22,15,1,31]
        #r_c = [22,5,4,31]
        #r_c = [22,14,24,32]
        #r_c = [22,14,4,29]
        #r_c = [22,5,4,31]
        #           k = 3
        #r_c = []
        #           k = 6
        #r_c = [22,14,4,56,46,55]
        #           k = 7
        #r_c = []
        #           k = 8
        #r_c = []


    p('')
    print('r_c:')
    print(r_c)
    p('')


    # do k means clustering using original data
    p('doing  k means clustering using original data')
    end_mk, iter_n, bi_l, grps, vhm, dun_o = k_means_processor(np_utk_data, km, rc=r_c)
    p('')
    p('')
    # do k means clustering using dimension reduced data data
    p('do k means clustering using dimension reduced data')
    end_mk2, iter2, bi_l2, grps2, vhm2, dun_z = k_means_processor(z_array, km, rc=r_c)
    p('')
    p('')
    p('do k means clustering using dimension reduced data using first 2 PC\'s')
    end_mk3, iter3, bi_l3, grps3, vhm3, dun_z2 = k_means_processor(z_array2, km, rc=r_c)
    p('')
    p('')



    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------

    # -------------------------------------------- Part 2-13 ----------------------------------------------------------
    # attempt to perform expectation maximization
    print('run number', run_cnt)
    run_cnt = run_cnt + 1
    p('Performing Expectation Maximization.......')
    mnew, hnew, em_itera = expectation_maximization(z_array2, end_mk3, bi_l3)
    uem, sem, vhem = numpy.linalg.svd(mnew, full_matrices=True, compute_uv=True)
    # make a grouping
    gbb = get_EM_grouping(hnew, km)
    grps4 = create_group_l(list(gbb.tolist()), km)

    em_mid = min_intercluster_distance(z_array2, grps4)
    em_mxid = max_intracluster_distance(z_array2, grps4)
    p('The number of interations: {:d}'.format(em_itera))
    p('The min intercluster distance is : {:f}'.format(em_mid))
    p('The max intracluster distance is : {:f}'.format(em_mxid))
    dun_em = em_mid / em_mxid
    p('the dun index is for k-means of original data is: {:f}'.format(dun_em))

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

if random_ave > 1:
    quit(0)

show_group = True

utk_names = create_group_names(grps, s_name)
pcak_names = create_group_names(grps2, s_name)
pca2_names = create_group_names(grps3, s_name)
pcaem_names = create_group_names(grps4, s_name)

# ----------------Grouping Display-----------------------------------------------------
if show_group:
    p('----------------------------------------------------------------------------------')
    p('Original Data Clustering:')
    #show_grouping(grps)
    print('Iterations: {:d}'.format(iter_n))
    show_grouping(utk_names)
    p('----------------------------------------------------------------------------------')
    p('')
    p('----------------------------------------------------------------------------------')
    p('PCA  Data Clustering:')
    #show_grouping(grps2)
    print('Iterations: {:d}'.format(iter2))
    show_grouping(pcak_names)
    p('----------------------------------------------------------------------------------')
    p('')
    p('----------------------------------------------------------------------------------')
    p('First 2 PC\'s Clustering:')
    #show_grouping(grps3)
    print('Iterations: {:d}'.format(iter3))
    show_grouping(pca2_names)
    p('----------------------------------------------------------------------------------')
    p('')
    p('----------------------------------------------------------------------------------')
    p('EM data Clustering:')
    #show_grouping(grps4)
    print('Iterations: {:d}'.format(em_itera))
    show_grouping(pcaem_names)
    p('----------------------------------------------------------------------------------')
    p('')

# ----------------------------------------------------------------------

vm = numpy.transpose(vhm)
vem = numpy.transpose(vhem)
vm2 = numpy.transpose(vhm2)
vm3 = numpy.transpose(vhm3)

Wm = vm[:, 0:k]

WTm = numpy.transpose(Wm)

m_stats2 = get_basic_stats(end_mk2)
m_stats3 = get_basic_stats(end_mk3)

uem, sem, vhem = numpy.linalg.svd(mnew, full_matrices=True, compute_uv=True)
Wem = vhem[0:k]
Wtem = numpy.transpose(Wem)
m_em_stat = get_basic_stats(mnew)

mid_points = project_mid_points(end_mk, vm, km)
mp2 = project_mid_points(end_mk2, vm2, km)
mp3 = project_mid_points(end_mk3, vm3, km)
emp = project_mid_points(mnew, vem, km)


colors_a = [[1, 0, 0],      # 0-red
            [0, 1, 0],      # 1-blue
            [0, 0, 1],      # 2-green
            [0, 0, 0],      # 3-black
            [1, 1, 0],      # 4-purp
            [1, 0, 1],      # 5
            [0, 1, 1],      # 6-yellow
            [.5, .5, .5],   # 7
            [.5, .2, .1],   # 8
            [.12, .2, .8]]  # 9

groups = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

legend_titles = ['Group 1',
                 'Group 2',
                 'Group 3',
                 'Group 4']

r_l = make_label_list(30)


show_graphs = True

if show_graphs:
#if len(sys.argv) == 1:

    if data_type == 1:
        k_cluster_title = 'Clustering {:d} groups for original data, dun = {:.2f}'.format(km, dun_o)
    else:
        k_cluster_title = 'K means with {:d} groups for original data, dun = {:.2f}'.format(km, dun_o)
    #z_scatter_plot(mid_points, groups, show_it=True)
    k_cluster_scatter_plot(z_array, s_name, mid_points, groups, colors=colors_a, b_list=bi_l, show_center=False,
                           title=k_cluster_title, legend=legend_titles, last=False, show_it=True)

    if data_type == 1:
        titlea = 'Clusters with PCA, {:d} k vectors, dun is {:.2f}'.format(km, dun_z)
    else:
        titlea = 'Clusters using PCA and {:d} k vectors dun is {:.2f}'.format(km, dun_z)
    k_cluster_scatter_plot(z_array, s_name, mp2, groups, colors=colors_a, b_list=bi_l2, show_center=False,
                           title=titlea, legend=legend_titles, last=False, show_it=True)

    if data_type == 1:
        title2 = 'Clusters w/ PCA, 1st 2 PC\'s,  dun {:.2f}'.format(dun_z2)
    else:
        title2 = 'Clusters using PCA with 1st 2 PC\'s,  dun {:.2f}'.format(dun_z2)
    k_cluster_scatter_plot(z_array2, s_name, mp3, groups, colors=colors_a, b_list=bi_l3, show_center=False,
                           title=title2, legend=legend_titles, last=False, show_it=True)

    if data_type == 1:
        title = 'Clusters w/PCA & EM, {:d} groups, dun: {:f}'.format(ko, dun_em)
    else:
        title = 'Clusters with PCA & EM, {:d} groups, dun: {:f}'.format(ko, dun_em)
    k_cluster_scatter_plot(z_array2, s_name, emp, groups, colors=colors_a, b_list=gbb, show_center=False,
                           title=title, legend=legend_titles, last=True, show_it=True, em=True, hnew=hnew)

